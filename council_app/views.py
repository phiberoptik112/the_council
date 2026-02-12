import json
import time

from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, DetailView, TemplateView, CreateView
from django.views import View
from django.contrib import messages
from django.http import JsonResponse, StreamingHttpResponse
from django.db.models import Count, Avg, Q
from django.utils import timezone
from django.urls import reverse

from .models import (
    Query, Response, Vote, ModelConfig, QueryTag, QueryEvent,
    CreativeProject, ChurnIteration, IterationFeedback, ChurnConfig,
    ReportKnowledgeBase, ReportOutline, ReportSection,
    PDFDocument, PDFPage,
)
from .forms import (
    QueryForm, ModelConfigForm, AddModelForm,
    CreativeProjectForm, SubmitContentForm, BranchForm,
    BatchChurnForm, TriggerChurnForm, ChurnSettingsForm,
    ReportKnowledgeBaseForm, ReportOutlineForm, SectionReviewForm,
    PDFUploadForm,
)
from .utils import get_system_info


class SubmitView(TemplateView):
    """Main page for submitting new queries"""
    template_name = 'council_app/submit.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = QueryForm()
        context['models'] = ModelConfig.objects.filter(is_active=True)
        context['recent_queries'] = Query.objects.all()[:5]
        context['system_info'] = get_system_info()
        return context
    
    def post(self, request):
        form = QueryForm(request.POST)
        models = request.POST.getlist('models')
        
        if not models or len(models) < 2:
            messages.error(request, 'Please select at least 2 models for the council.')
            return redirect('council_app:submit')
        
        if form.is_valid():
            query = form.save()
            
            # Queue the council task
            from django_q.tasks import async_task
            async_task(
                'council_app.tasks.run_council_query',
                query.id,
                [int(m) for m in models],
                task_name=f'council_query_{query.id}'
            )
            
            messages.success(request, 'Query submitted! The council is deliberating...')
            return redirect('council_app:results', pk=query.id)
        
        return render(request, self.template_name, {
            'form': form,
            'models': ModelConfig.objects.filter(is_active=True),
            'recent_queries': Query.objects.all()[:5],
        })


class ResultsView(DetailView):
    """View query results"""
    model = Query
    template_name = 'council_app/results.html'
    context_object_name = 'query'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        query = self.object
        
        # Get responses with score percentages
        responses = list(query.responses.select_related('model').all())
        max_score = max((r.score for r in responses), default=1) or 1
        for response in responses:
            response.score_percent = (response.score / max_score) * 100
        
        context['responses'] = responses
        context['votes'] = query.votes.select_related('voter_model').all()
        
        # Get events for this query (for error display)
        context['events'] = QueryEvent.objects.filter(query=query).order_by('created_at')
        
        # Build error details from events
        if query.status == 'error':
            error_events = QueryEvent.objects.filter(
                query=query,
                event_type__in=['error', 'timeout', 'query_error']
            )
            model_errors = {}
            for event in error_events:
                if event.model_name:
                    model_errors[event.model_name] = {
                        'type': event.event_type,
                        'message': event.message,
                        'raw_data': event.raw_data
                    }
            context['model_errors'] = model_errors
            
            # Generate troubleshooting hints based on error types
            hints = []
            error_types = set(e.event_type for e in error_events)
            
            if 'timeout' in error_types:
                hints.append("Some models timed out. Consider increasing model timeout in Settings > Models.")
                hints.append("Smaller models (tinyllama, phi3:mini) respond faster on limited hardware.")
            
            if any('404' in str(e.raw_data) or '404' in e.message for e in error_events):
                hints.append("Model not found in Ollama. Run 'ollama pull <model_name>' to download.")
            
            if any('connection' in e.message.lower() for e in error_events):
                hints.append("Cannot connect to Ollama. Check if it's running with 'systemctl status ollama'.")
            
            if not hints:
                hints.append("Check the event log below for detailed error information.")
                hints.append("Try with fewer or smaller models.")
            
            context['troubleshooting_hints'] = hints
        
        return context


def query_status(request, pk):
    """HTMX endpoint for polling query status"""
    query = get_object_or_404(Query, pk=pk)
    
    # Get events for this query
    events = QueryEvent.objects.filter(query=query).order_by('-created_at')[:20]
    
    # Build model status from events
    model_status = {}
    for event in reversed(list(events)):
        if event.model_name:
            model_status[event.model_name] = {
                'status': event.event_type,
                'message': event.message,
                'raw_data': event.raw_data
            }
    
    return render(request, 'council_app/partials/status.html', {
        'query': query,
        'events': events,
        'model_status': model_status
    })


def query_events_stream(request, pk):
    """
    Server-Sent Events endpoint for real-time query updates.
    
    Streams query events as they occur, allowing the frontend to show
    real-time progress without polling.
    
    Includes a maximum stream lifetime (10 minutes) to prevent stale
    connections from accumulating when clients disconnect unexpectedly
    (e.g. laptop going to sleep).
    """
    MAX_STREAM_SECONDS = 600  # 10 minutes maximum stream lifetime

    def event_stream():
        last_event_id = 0
        heartbeat_interval = 15  # seconds
        last_heartbeat = time.time()
        stream_start = time.time()
        
        while True:
            try:
                # Check if the stream has exceeded its maximum lifetime
                if time.time() - stream_start > MAX_STREAM_SECONDS:
                    timeout_data = {
                        'type': 'stream_end',
                        'status': 'timeout',
                        'error_message': 'Stream timed out. Refresh the page to reconnect.'
                    }
                    yield f"data: {json.dumps(timeout_data)}\n\n"
                    break

                # Get the query
                query = Query.objects.get(pk=pk)
                
                # Get new events since last check
                events = QueryEvent.objects.filter(
                    query_id=pk,
                    id__gt=last_event_id
                ).order_by('id')
                
                for event in events:
                    data = {
                        'id': event.id,
                        'type': event.event_type,
                        'model': event.model_name,
                        'message': event.message,
                        'raw': event.raw_data,
                        'timestamp': event.created_at.isoformat()
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    last_event_id = event.id
                
                # Check if query is complete or errored
                if query.status in ['complete', 'error']:
                    # Send final status event
                    final_data = {
                        'type': 'stream_end',
                        'status': query.status,
                        'error_message': query.error_message if query.status == 'error' else None
                    }
                    yield f"data: {json.dumps(final_data)}\n\n"
                    break
                
                # Send heartbeat to keep connection alive
                current_time = time.time()
                if current_time - last_heartbeat > heartbeat_interval:
                    yield f": heartbeat\n\n"
                    last_heartbeat = current_time
                
                # Brief sleep to prevent tight loop
                time.sleep(1)
                
            except Query.DoesNotExist:
                error_data = {'type': 'error', 'message': 'Query not found'}
                yield f"data: {json.dumps(error_data)}\n\n"
                break
            except Exception as e:
                error_data = {'type': 'error', 'message': str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"
                break
    
    response = StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    return response


def query_events_json(request, pk):
    """
    JSON endpoint to get all events for a query (for initial load or fallback).
    """
    query = get_object_or_404(Query, pk=pk)
    events = QueryEvent.objects.filter(query=query).order_by('created_at')
    
    events_data = [{
        'id': event.id,
        'type': event.event_type,
        'model': event.model_name,
        'message': event.message,
        'raw': event.raw_data,
        'timestamp': event.created_at.isoformat()
    } for event in events]
    
    return JsonResponse({
        'query_id': pk,
        'status': query.status,
        'events': events_data
    })


class HistoryView(ListView):
    """Browse query history"""
    model = Query
    template_name = 'council_app/history.html'
    context_object_name = 'queries'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = Query.objects.select_related('winner', 'winner__model').prefetch_related('responses', 'votes')
        
        # Search filter
        search = self.request.GET.get('search', '').strip()
        if search:
            queryset = queryset.filter(prompt__icontains=search)
        
        # Status filter
        status = self.request.GET.get('status', '').strip()
        if status:
            queryset = queryset.filter(status=status)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['search_query'] = self.request.GET.get('search', '')
        context['status_filter'] = self.request.GET.get('status', '')
        return context


class AnalyticsView(TemplateView):
    """Analytics dashboard"""
    template_name = 'council_app/analytics.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Overall stats
        context['total_queries'] = Query.objects.count()
        context['completed_queries'] = Query.objects.filter(status='complete').count()
        context['active_models'] = ModelConfig.objects.filter(is_active=True).count()
        
        # Average response time
        avg_time = Response.objects.aggregate(avg=Avg('response_time'))
        context['avg_response_time'] = avg_time['avg'] or 0
        
        # Model statistics
        context['model_stats'] = ModelConfig.objects.filter(
            total_queries__gt=0
        ).order_by('-win_rate', '-total_wins')
        
        # Mode distribution
        mode_counts = Query.objects.filter(status='complete').values('council_mode').annotate(
            count=Count('id')
        )
        total_mode_queries = sum(m['count'] for m in mode_counts) or 1
        for mode in mode_counts:
            mode['percent'] = (mode['count'] / total_mode_queries) * 100
        context['mode_stats'] = mode_counts
        
        return context


class ModelsView(TemplateView):
    """Model configuration page"""
    template_name = 'council_app/models.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['models'] = ModelConfig.objects.all().order_by('-is_active', 'name')
        context['add_form'] = AddModelForm()
        return context


def add_model(request):
    """Add a new model"""
    if request.method == 'POST':
        form = AddModelForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['model_name']
            ModelConfig.objects.create(
                name=name,
                display_name=name.split(':')[0].title() if ':' in name else name.title()
            )
            messages.success(request, f'Model "{name}" added successfully!')
        else:
            for error in form.errors.values():
                messages.error(request, error[0])
    
    return redirect('council_app:models')


def toggle_model(request, pk):
    """Toggle model active state"""
    if request.method == 'POST':
        model = get_object_or_404(ModelConfig, pk=pk)
        model.is_active = not model.is_active
        model.save()
        
        status = 'enabled' if model.is_active else 'disabled'
        messages.success(request, f'Model "{model.name}" {status}.')
    
    return redirect('council_app:models')


def delete_model(request, pk):
    """Delete a model"""
    if request.method == 'POST':
        model = get_object_or_404(ModelConfig, pk=pk)
        name = model.name
        model.delete()
        messages.success(request, f'Model "{name}" deleted.')
    
    return redirect('council_app:models')


def update_model_timeout(request, pk):
    """Update a model's timeout setting"""
    if request.method == 'POST':
        model = get_object_or_404(ModelConfig, pk=pk)
        timeout_str = request.POST.get('timeout', '').strip()
        
        if timeout_str:
            try:
                timeout = int(timeout_str)
                if 30 <= timeout <= 1800:  # 30 seconds to 30 minutes
                    model.timeout = timeout
                    model.save()
                    messages.success(request, f'Timeout for "{model.name}" set to {timeout}s.')
                else:
                    messages.error(request, 'Timeout must be between 30 and 1800 seconds.')
            except ValueError:
                messages.error(request, 'Invalid timeout value.')
        else:
            # Clear the timeout (use default)
            model.timeout = None
            model.save()
            messages.success(request, f'Timeout for "{model.name}" reset to default.')
    
    return redirect('council_app:models')


# =============================================================================
# CHURN MACHINE VIEWS
# =============================================================================

class ChurnProjectListView(ListView):
    """List all creative writing projects"""
    model = CreativeProject
    template_name = 'council_app/churn/project_list.html'
    context_object_name = 'projects'
    paginate_by = 12
    
    def get_queryset(self):
        queryset = CreativeProject.objects.prefetch_related('iterations').all()
        
        # Search filter
        search = self.request.GET.get('search', '').strip()
        if search:
            queryset = queryset.filter(
                Q(title__icontains=search) | Q(description__icontains=search)
            )
        
        # Content type filter
        content_type = self.request.GET.get('content_type', '').strip()
        if content_type:
            queryset = queryset.filter(content_type=content_type)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['search_query'] = self.request.GET.get('search', '')
        context['content_type_filter'] = self.request.GET.get('content_type', '')
        context['content_types'] = CreativeProject.ContentType.choices
        return context


class CreateProjectView(TemplateView):
    """Create a new creative writing project"""
    template_name = 'council_app/churn/create_project.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = CreativeProjectForm()
        context['models'] = ModelConfig.objects.filter(is_active=True)
        return context
    
    def post(self, request):
        form = CreativeProjectForm(request.POST)
        models = request.POST.getlist('models')
        
        if not models or len(models) < 2:
            messages.error(request, 'Please select at least 2 models for the council.')
            return redirect('council_app:churn_create')
        
        if form.is_valid():
            project = form.save()
            # Set default models
            project.default_models.set(ModelConfig.objects.filter(pk__in=models))
            
            # Route to report setup if content type is report
            if project.content_type == CreativeProject.ContentType.REPORT:
                messages.success(request, f'Project "{project.title}" created! Now set up your report outline.')
                return redirect('council_app:report_setup', pk=project.id)
            
            messages.success(request, f'Project "{project.title}" created! Now add your content.')
            return redirect('council_app:churn_submit', pk=project.id)
        
        return render(request, self.template_name, {
            'form': form,
            'models': ModelConfig.objects.filter(is_active=True),
        })


class ProjectDetailView(DetailView):
    """View a creative project with its iteration tree"""
    model = CreativeProject
    template_name = 'council_app/churn/project_detail.html'
    context_object_name = 'project'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        project = self.object
        
        # Get all iterations organized by tree structure
        context['root_iteration'] = project.root_iteration
        context['all_iterations'] = project.iterations.select_related('parent').prefetch_related('children', 'feedback')
        context['iteration_tree'] = project.get_iteration_tree()
        latest_iteration = project.latest_iteration
        context['latest_iteration'] = latest_iteration
        context['latest_iteration_has_feedback'] = (
            latest_iteration
            and latest_iteration.status == 'complete'
            and IterationFeedback.objects.filter(iteration=latest_iteration).exists()
        )
        
        # Get branch form for quick branching
        context['branch_form'] = BranchForm()
        context['trigger_form'] = TriggerChurnForm()
        context['batch_form'] = BatchChurnForm()
        
        return context


class ChurnSettingsView(View):
    """View for Churn Machine performance settings"""
    template_name = 'council_app/churn/churn_settings.html'

    def get(self, request):
        config = ChurnConfig.get_instance()
        form = ChurnSettingsForm(instance=config)
        return render(request, self.template_name, {
            'form': form,
            'config': config,
        })

    def post(self, request):
        config = ChurnConfig.get_instance()
        form = ChurnSettingsForm(request.POST, instance=config)
        if form.is_valid():
            form.save()
            messages.success(request, 'Churn settings saved successfully.')
            return redirect('council_app:churn_list')
        return render(request, self.template_name, {
            'form': form,
            'config': config,
        })


class IterationDetailView(DetailView):
    """View a specific iteration with diff and feedback"""
    model = ChurnIteration
    template_name = 'council_app/churn/iteration_detail.html'
    context_object_name = 'iteration'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        iteration = self.object
        
        # Get parent for comparison
        context['parent'] = iteration.parent
        context['branch_path'] = iteration.get_branch_path()
        context['root_iteration'] = iteration.get_branch_path()[0] if iteration.get_branch_path() else None
        context['siblings'] = iteration.get_siblings()
        context['children'] = iteration.children.all()
        
        # Get feedback if available
        try:
            context['feedback'] = iteration.feedback
        except IterationFeedback.DoesNotExist:
            context['feedback'] = None
        
        # Forms
        context['branch_form'] = BranchForm()
        context['trigger_form'] = TriggerChurnForm(initial={
            'churn_type': iteration.project.default_churn_type
        })
        context['batch_form'] = BatchChurnForm(initial={
            'churn_type': iteration.project.default_churn_type
        })
        
        # Parse diff for display
        if iteration.content_diff:
            context['diff_lines'] = parse_diff_for_display(iteration.content_diff)
        
        # Add debug info if available
        if iteration.debug_info:
            context['debug_info'] = iteration.debug_info
        
        return context


class SubmitIterationView(TemplateView):
    """Submit content for the first iteration of a project"""
    template_name = 'council_app/churn/submit_content.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        project = get_object_or_404(CreativeProject, pk=self.kwargs['pk'])
        context['project'] = project
        context['form'] = SubmitContentForm(initial={
            'churn_type': project.default_churn_type
        })
        context['models'] = ModelConfig.objects.filter(is_active=True)
        
        # Pre-select project's default models
        context['default_model_ids'] = list(project.default_models.values_list('id', flat=True))
        
        return context
    
    def post(self, request, pk):
        project = get_object_or_404(CreativeProject, pk=pk)
        form = SubmitContentForm(request.POST)
        models = request.POST.getlist('models')
        
        if not models or len(models) < 2:
            messages.error(request, 'Please select at least 2 models for the council.')
            return redirect('council_app:churn_submit', pk=pk)
        
        if form.is_valid():
            # Create the root iteration
            iteration = ChurnIteration.objects.create(
                project=project,
                content=form.cleaned_data['content'],
                churn_type=form.cleaned_data['churn_type'],
                iteration_number=1,
                status=ChurnIteration.Status.PENDING
            )
            iteration.models_used.set(ModelConfig.objects.filter(pk__in=models))
            
            # Queue the churn task
            from django_q.tasks import async_task
            async_task(
                'council_app.tasks.run_churn_iteration',
                iteration.id,
                [int(m) for m in models],
                form.cleaned_data.get('context', ''),
                task_name=f'churn_iteration_{iteration.id}'
            )
            
            messages.success(request, 'Content submitted! The council is churning...')
            return redirect('council_app:churn_iteration', pk=iteration.id)
        
        return render(request, self.template_name, {
            'project': project,
            'form': form,
            'models': ModelConfig.objects.filter(is_active=True),
            'default_model_ids': list(project.default_models.values_list('id', flat=True)),
        })


class TriggerChurnView(View):
    """Trigger a new churn iteration from an existing one"""
    
    def post(self, request, pk):
        parent_iteration = get_object_or_404(ChurnIteration, pk=pk)
        form = TriggerChurnForm(request.POST)
        models = request.POST.getlist('models')
        
        if not models or len(models) < 2:
            messages.error(request, 'Please select at least 2 models for the council.')
            return redirect('council_app:churn_iteration', pk=pk)
        
        if form.is_valid():
            # Determine starting content
            use_suggested = form.cleaned_data.get('use_suggested_content', True)
            try:
                feedback = parent_iteration.feedback
                if use_suggested and feedback and feedback.suggested_content:
                    starting_content = feedback.suggested_content
                else:
                    starting_content = parent_iteration.content
            except IterationFeedback.DoesNotExist:
                starting_content = parent_iteration.content
            
            # Create new iteration
            iteration = ChurnIteration.objects.create(
                project=parent_iteration.project,
                parent=parent_iteration,
                content=starting_content,
                churn_type=form.cleaned_data['churn_type'],
                iteration_number=parent_iteration.iteration_number + 1,
                status=ChurnIteration.Status.PENDING
            )
            iteration.models_used.set(ModelConfig.objects.filter(pk__in=models))
            
            # Queue the churn task
            from django_q.tasks import async_task
            async_task(
                'council_app.tasks.run_churn_iteration',
                iteration.id,
                [int(m) for m in models],
                '',  # context
                task_name=f'churn_iteration_{iteration.id}'
            )
            
            messages.success(request, 'New iteration started! The council is churning...')
            return redirect('council_app:churn_iteration', pk=iteration.id)
        
        messages.error(request, 'Invalid form submission.')
        return redirect('council_app:churn_iteration', pk=pk)


class RetryIterationView(View):
    """Retry a failed iteration"""
    
    def post(self, request, pk):
        iteration = get_object_or_404(ChurnIteration, pk=pk)
        
        # Only allow retry on error status
        if iteration.status != ChurnIteration.Status.ERROR:
            messages.error(request, 'Can only retry failed iterations.')
            return redirect('council_app:churn_iteration', pk=pk)
        
        # Optional: Allow model selection override
        models = request.POST.getlist('models')
        if models:
            # Validate models exist and are active
            model_objs = ModelConfig.objects.filter(pk__in=models, is_active=True)
            if model_objs.count() < 2:
                messages.error(request, 'Please select at least 2 active models for the retry.')
                return redirect('council_app:churn_iteration', pk=pk)
            iteration.models_used.set(model_objs)
        else:
            # Use existing models
            if iteration.models_used.filter(is_active=True).count() < 2:
                messages.error(request, 'Please select at least 2 active models for the retry.')
                return redirect('council_app:churn_iteration', pk=pk)
        
        # Reset iteration state
        iteration.status = ChurnIteration.Status.PENDING
        iteration.error_message = ''
        iteration.completed_at = None  # Clear completion timestamp if exists
        iteration.cancel_requested = False
        iteration.save()
        
        # Delete incomplete feedback if exists
        try:
            iteration.feedback.delete()
        except IterationFeedback.DoesNotExist:
            pass
        
        # Requeue task
        from django_q.tasks import async_task
        model_ids = list(iteration.models_used.values_list('id', flat=True))
        async_task(
            'council_app.tasks.run_churn_iteration',
            iteration.id,
            model_ids,
            '',  # context
            task_name=f'churn_iteration_{iteration.id}_retry'
        )
        
        messages.success(request, 'Iteration retry queued! The council will reprocess this iteration.')
        return redirect('council_app:churn_iteration', pk=pk)


class StopChurnView(View):
    """Request cancellation of an in-progress churn iteration."""
    
    def post(self, request, pk):
        iteration = get_object_or_404(ChurnIteration, pk=pk)
        
        if iteration.status != ChurnIteration.Status.PROCESSING:
            messages.warning(request, 'Iteration is not processing. Nothing to stop.')
            return redirect('council_app:churn_iteration', pk=pk)
        
        iteration.cancel_requested = True
        iteration.save(update_fields=['cancel_requested'])
        
        messages.success(request, 'Stop requested. The iteration will stop at the next stage boundary.')
        return redirect('council_app:churn_iteration', pk=pk)


class BranchView(View):
    """Create a branch from an iteration"""
    
    def post(self, request, pk):
        parent_iteration = get_object_or_404(ChurnIteration, pk=pk)
        form = BranchForm(request.POST)
        
        if form.is_valid():
            # Determine starting content
            use_suggested = form.cleaned_data.get('use_suggested_content', True)
            try:
                feedback = parent_iteration.feedback
                if use_suggested and feedback and feedback.suggested_content:
                    starting_content = feedback.suggested_content
                else:
                    starting_content = parent_iteration.content
            except IterationFeedback.DoesNotExist:
                starting_content = parent_iteration.content
            
            # Create branch iteration (uses EXPLORE type by default)
            iteration = ChurnIteration.objects.create(
                project=parent_iteration.project,
                parent=parent_iteration,
                branch_name=form.cleaned_data['branch_name'],
                content=starting_content,
                churn_type=CreativeProject.ChurnType.EXPLORE,
                iteration_number=1,  # Reset for new branch
                status=ChurnIteration.Status.PENDING
            )
            
            # Copy models from parent or use project defaults
            if parent_iteration.models_used.exists():
                iteration.models_used.set(parent_iteration.models_used.all())
            else:
                iteration.models_used.set(parent_iteration.project.default_models.all())
            
            # Queue the churn task with direction
            from django_q.tasks import async_task
            model_ids = list(iteration.models_used.values_list('id', flat=True))
            direction = form.cleaned_data.get('direction', '')
            
            async_task(
                'council_app.tasks.run_churn_iteration',
                iteration.id,
                model_ids,
                direction,  # Use direction as context for explore mode
                task_name=f'churn_branch_{iteration.id}'
            )
            
            messages.success(request, f'Branch "{iteration.branch_name}" created!')
            return redirect('council_app:churn_iteration', pk=iteration.id)
        
        messages.error(request, 'Invalid branch name.')
        return redirect('council_app:churn_iteration', pk=pk)


class CompareView(TemplateView):
    """Side-by-side comparison of two iterations"""
    template_name = 'council_app/churn/compare.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        iteration1 = get_object_or_404(ChurnIteration, pk=self.kwargs['pk1'])
        iteration2 = get_object_or_404(ChurnIteration, pk=self.kwargs['pk2'])
        
        context['iteration1'] = iteration1
        context['iteration2'] = iteration2
        context['project'] = iteration1.project
        
        # Generate diff between the two
        from .churn import ChurnEngine
        engine = ChurnEngine(models=[])  # Just using for diff generation
        context['diff'] = engine.generate_unified_diff(iteration1.content, iteration2.content)
        context['diff_lines'] = parse_diff_for_display(context['diff'])
        context['word_diff'] = engine.generate_word_diff(iteration1.content, iteration2.content)
        
        return context


def iteration_status(request, pk):
    """HTMX endpoint for polling iteration status"""
    iteration = get_object_or_404(ChurnIteration, pk=pk)
    context = {
        'iteration': iteration
    }
    # Add debug_info if available
    if iteration.debug_info:
        context['debug_info'] = iteration.debug_info
    return render(request, 'council_app/churn/partials/iteration_status.html', context)


def parse_diff_for_display(diff_text):
    """Parse unified diff into structured format for display"""
    if not diff_text:
        return []
    
    lines = []
    for line in diff_text.split('\n'):
        if line.startswith('+++') or line.startswith('---'):
            lines.append({'type': 'header', 'content': line})
        elif line.startswith('@@'):
            lines.append({'type': 'hunk', 'content': line})
        elif line.startswith('+'):
            lines.append({'type': 'addition', 'content': line[1:]})
        elif line.startswith('-'):
            lines.append({'type': 'deletion', 'content': line[1:]})
        else:
            lines.append({'type': 'context', 'content': line})
    
    return lines


class BatchChurnView(View):
    """Run multiple churn iterations automatically"""
    
    def post(self, request, pk):
        iteration = get_object_or_404(ChurnIteration, pk=pk)
        form = BatchChurnForm(request.POST)
        models = request.POST.getlist('models')
        
        if not models or len(models) < 2:
            messages.error(request, 'Please select at least 2 models for the council.')
            return redirect('council_app:churn_iteration', pk=pk)
        
        if form.is_valid():
            num_iterations = form.cleaned_data['num_iterations']
            auto_apply = form.cleaned_data['auto_apply']
            churn_type = form.cleaned_data['churn_type']
            
            # Queue batch task
            from django_q.tasks import async_task
            async_task(
                'council_app.tasks.run_batch_churn',
                iteration.project.id,
                iteration.id,
                num_iterations,
                [int(m) for m in models],
                auto_apply,
                churn_type,
                task_name=f'batch_churn_{iteration.project.id}'
            )
            
            messages.success(request, f'Batch processing started! Running {num_iterations} iterations...')
            return redirect('council_app:churn_project', pk=iteration.project.id)
        
        messages.error(request, 'Invalid batch settings.')
        return redirect('council_app:churn_iteration', pk=pk)


# =============================================================================
# TECHNICAL REPORT REVIEWER VIEWS
# =============================================================================

class KnowledgeBaseListView(ListView):
    """List all report knowledgebases"""
    model = ReportKnowledgeBase
    template_name = 'council_app/churn/kb_list.html'
    context_object_name = 'knowledgebases'
    paginate_by = 12


class KnowledgeBaseCreateView(TemplateView):
    """Create a new knowledgebase"""
    template_name = 'council_app/churn/kb_form.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = ReportKnowledgeBaseForm()
        context['is_edit'] = False
        return context
    
    def post(self, request):
        form = ReportKnowledgeBaseForm(request.POST)
        if form.is_valid():
            kb = form.save()
            messages.success(request, f'Knowledge base "{kb.name}" created!')
            return redirect('council_app:kb_detail', pk=kb.id)
        
        return render(request, self.template_name, {
            'form': form,
            'is_edit': False,
        })


class KnowledgeBaseDetailView(TemplateView):
    """View/edit a knowledgebase"""
    template_name = 'council_app/churn/kb_form.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        kb = get_object_or_404(ReportKnowledgeBase, pk=self.kwargs['pk'])
        context['kb'] = kb
        context['form'] = ReportKnowledgeBaseForm(instance=kb)
        context['is_edit'] = True
        # Show reports using this KB
        context['linked_outlines'] = kb.report_outlines.select_related('project').all()
        return context
    
    def post(self, request, pk):
        kb = get_object_or_404(ReportKnowledgeBase, pk=pk)
        form = ReportKnowledgeBaseForm(request.POST, instance=kb)
        if form.is_valid():
            form.save()
            messages.success(request, f'Knowledge base "{kb.name}" updated!')
            return redirect('council_app:kb_detail', pk=kb.id)
        
        return render(request, self.template_name, {
            'kb': kb,
            'form': form,
            'is_edit': True,
            'linked_outlines': kb.report_outlines.select_related('project').all(),
        })


class ReportSetupView(TemplateView):
    """Set up a report outline for a project"""
    template_name = 'council_app/churn/report_setup.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        project = get_object_or_404(CreativeProject, pk=self.kwargs['pk'])
        context['project'] = project
        context['form'] = ReportOutlineForm()
        context['models'] = ModelConfig.objects.filter(is_active=True)
        context['default_model_ids'] = list(project.default_models.values_list('id', flat=True))
        
        # Check if outline already exists
        try:
            context['existing_outline'] = project.report_outline
        except ReportOutline.DoesNotExist:
            context['existing_outline'] = None
        
        return context
    
    def post(self, request, pk):
        project = get_object_or_404(CreativeProject, pk=pk)
        form = ReportOutlineForm(request.POST)
        
        if form.is_valid():
            # Parse the outline
            from .report_churn import OutlineParser
            parser = OutlineParser()
            parsed = parser.parse(form.cleaned_data['raw_outline'])
            
            if not parsed:
                messages.error(request, 'Could not parse any sections from the outline. Check your formatting.')
                return redirect('council_app:report_setup', pk=pk)
            
            # Create or update the ReportOutline
            outline, created = ReportOutline.objects.update_or_create(
                project=project,
                defaults={
                    'raw_outline': form.cleaned_data['raw_outline'],
                    'parsed_sections': parsed,
                    'knowledgebase': form.cleaned_data.get('knowledgebase'),
                    'report_type': form.cleaned_data['report_type'],
                    'target_audience': form.cleaned_data.get('target_audience', ''),
                    'processing_mode': form.cleaned_data['processing_mode'],
                }
            )
            
            # Create ReportSection objects for each parsed section
            for section_data in parsed:
                ReportSection.objects.get_or_create(
                    report_outline=outline,
                    section_id=section_data['id'],
                    defaults={
                        'section_title': section_data['title'],
                        'original_content': section_data.get('content', ''),
                        'current_content': section_data.get('content', ''),
                        'order': section_data.get('order', 0),
                    }
                )
            
            action = 'updated' if not created else 'created'
            messages.success(
                request,
                f'Report outline {action} with {len(parsed)} sections!'
            )
            return redirect('council_app:report_detail', pk=project.id)
        
        return render(request, self.template_name, {
            'project': project,
            'form': form,
            'models': ModelConfig.objects.filter(is_active=True),
            'default_model_ids': list(project.default_models.values_list('id', flat=True)),
        })


class ReportDetailView(DetailView):
    """Dashboard showing all sections with status/compliance scores"""
    model = CreativeProject
    template_name = 'council_app/churn/report_detail.html'
    context_object_name = 'project'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        project = self.object
        
        try:
            outline = project.report_outline
            context['outline'] = outline
            context['sections'] = outline.sections.all().order_by('order')
            context['progress_percent'] = outline.progress_percent
            context['total_sections'] = outline.sections.count()
            context['approved_sections'] = outline.sections.filter(
                status=ReportSection.Status.APPROVED
            ).count()
            context['review_sections'] = outline.sections.filter(
                status=ReportSection.Status.REVIEW
            ).count()
            context['pending_sections'] = outline.sections.filter(
                status__in=[
                    ReportSection.Status.PENDING,
                    ReportSection.Status.NEEDS_REVISION,
                ]
            ).count()
            context['in_progress_sections'] = outline.sections.filter(
                status=ReportSection.Status.IN_PROGRESS
            ).count()
        except ReportOutline.DoesNotExist:
            context['outline'] = None
            context['sections'] = []
        
        context['models'] = ModelConfig.objects.filter(is_active=True)
        context['default_model_ids'] = list(
            project.default_models.values_list('id', flat=True)
        )
        
        return context


class SectionDetailView(DetailView):
    """View a single report section with feedback and diff"""
    model = ReportSection
    template_name = 'council_app/churn/section_detail.html'
    context_object_name = 'section'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        section = self.object
        outline = section.report_outline
        project = outline.project
        
        context['project'] = project
        context['outline'] = outline
        context['review_form'] = SectionReviewForm()
        context['models'] = ModelConfig.objects.filter(is_active=True)
        context['default_model_ids'] = list(
            project.default_models.values_list('id', flat=True)
        )
        
        # Parse diff for display
        if section.content_diff:
            context['diff_lines'] = parse_diff_for_display(section.content_diff)
        
        # Debug info for live output display during processing
        if section.debug_info:
            context['debug_info'] = section.debug_info
        
        # Get feedback details from JSON
        feedback = section.council_feedback or {}
        context['synthesized_feedback'] = feedback.get('synthesized_feedback', '')
        context['processing_time'] = feedback.get('processing_time')
        
        # Navigation: previous and next sections
        all_sections = list(outline.sections.order_by('order'))
        current_idx = next(
            (i for i, s in enumerate(all_sections) if s.pk == section.pk), 0
        )
        context['prev_section'] = all_sections[current_idx - 1] if current_idx > 0 else None
        context['next_section'] = all_sections[current_idx + 1] if current_idx < len(all_sections) - 1 else None
        
        return context


class TriggerSectionChurnView(View):
    """Trigger council review for a single section"""
    
    def post(self, request, pk):
        section = get_object_or_404(ReportSection, pk=pk)
        models = request.POST.getlist('models')
        
        if not models or len(models) < 2:
            messages.error(request, 'Please select at least 2 models for the council.')
            return redirect('council_app:section_detail', pk=pk)
        
        # Queue the section churn task
        from django_q.tasks import async_task
        async_task(
            'council_app.tasks.run_report_section_churn',
            section.id,
            [int(m) for m in models],
            task_name=f'report_section_{section.id}'
        )
        
        messages.success(request, f'Section "{section.section_title}" submitted for review!')
        return redirect('council_app:section_detail', pk=pk)


class StopSectionChurnView(View):
    """Request cancellation of an in-progress report section churn."""
    
    def post(self, request, pk):
        section = get_object_or_404(ReportSection, pk=pk)
        
        if section.status != ReportSection.Status.IN_PROGRESS:
            messages.warning(request, 'Section is not in progress. Nothing to stop.')
            return redirect('council_app:section_detail', pk=pk)
        
        section.cancel_requested = True
        section.save(update_fields=['cancel_requested'])
        
        messages.success(request, 'Stop requested. The section will stop at the next stage boundary.')
        return redirect('council_app:section_detail', pk=pk)


class TriggerFullReportChurnView(View):
    """Trigger council review for all pending sections"""
    
    def post(self, request, pk):
        project = get_object_or_404(CreativeProject, pk=pk)
        models = request.POST.getlist('models')
        auto_advance = request.POST.get('auto_advance', '') == 'on'
        
        if not models or len(models) < 2:
            messages.error(request, 'Please select at least 2 models for the council.')
            return redirect('council_app:report_detail', pk=pk)
        
        try:
            outline = project.report_outline
        except ReportOutline.DoesNotExist:
            messages.error(request, 'No report outline found. Please set one up first.')
            return redirect('council_app:report_setup', pk=pk)
        
        # Queue the full report churn task
        from django_q.tasks import async_task
        async_task(
            'council_app.tasks.run_full_report_churn',
            outline.id,
            [int(m) for m in models],
            auto_advance,
            task_name=f'full_report_{outline.id}'
        )
        
        mode = 'all sections' if auto_advance else 'next pending section'
        messages.success(request, f'Report review started! Processing {mode}...')
        return redirect('council_app:report_detail', pk=pk)


class ApproveSectionView(View):
    """Handle section review actions: approve, revise, or edit"""
    
    def post(self, request, pk):
        section = get_object_or_404(ReportSection, pk=pk)
        form = SectionReviewForm(request.POST)
        
        if form.is_valid():
            action = form.cleaned_data['action']
            
            if action == 'approve':
                section.mark_approved()
                messages.success(request, f'Section "{section.section_title}" approved!')
            
            elif action == 'revise':
                section.mark_needs_revision()
                messages.info(request, f'Section "{section.section_title}" marked for revision.')
            
            elif action == 'edit':
                edited_content = form.cleaned_data.get('edited_content', '').strip()
                if edited_content:
                    section.current_content = edited_content
                    section.mark_approved()
                    messages.success(
                        request,
                        f'Section "{section.section_title}" updated and approved!'
                    )
                else:
                    messages.error(request, 'Please provide edited content when using "Edit Manually".')
                    return redirect('council_app:section_detail', pk=pk)
            
            return redirect('council_app:report_detail', pk=section.report_outline.project.id)
        
        messages.error(request, 'Invalid form submission.')
        return redirect('council_app:section_detail', pk=pk)


def section_status(request, pk):
    """HTMX endpoint for polling section processing status"""
    section = get_object_or_404(ReportSection, pk=pk)
    context = {'section': section}
    if section.debug_info:
        context['debug_info'] = section.debug_info
    return render(request, 'council_app/churn/partials/section_status.html', context)


# =============================================================================
# PDF TO MARKDOWN VIEWS
# =============================================================================

class PDFListView(ListView):
    """List all uploaded PDF documents"""
    model = PDFDocument
    template_name = 'council_app/churn/pdf_list.html'
    context_object_name = 'documents'
    paginate_by = 12


class PDFUploadView(TemplateView):
    """Upload a new PDF for markdown extraction"""
    template_name = 'council_app/churn/pdf_upload.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = PDFUploadForm()
        return context

    def post(self, request):
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save()

            # Get page count immediately so we can show it
            try:
                from .pdf_processor import PDFPageProcessor
                total_pages = PDFPageProcessor.extract_page_count(document.file.path)
                document.total_pages = total_pages
                document.save(update_fields=['total_pages'])
            except Exception as e:
                document.total_pages = 0
                document.save(update_fields=['total_pages'])

            messages.success(
                request,
                f'PDF "{document.title}" uploaded ({document.total_pages} pages). '
                f'Click "Start Processing" to begin extraction.'
            )
            return redirect('council_app:pdf_detail', pk=document.id)

        return render(request, self.template_name, {'form': form})


class PDFDetailView(DetailView):
    """View a PDF document with its pages and processing status"""
    model = PDFDocument
    template_name = 'council_app/churn/pdf_detail.html'
    context_object_name = 'document'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        document = self.object
        context['pages'] = document.pages.all().order_by('page_number')
        context['complete_pages'] = document.pages.filter(
            status=PDFPage.Status.COMPLETE
        ).count()
        context['error_pages'] = document.pages.filter(
            status=PDFPage.Status.ERROR
        ).count()
        context['models'] = ModelConfig.objects.filter(is_active=True)
        return context


class PDFProcessView(View):
    """Trigger PDF processing as a background task"""

    def post(self, request, pk):
        document = get_object_or_404(PDFDocument, pk=pk)

        if document.status == PDFDocument.Status.PROCESSING:
            messages.warning(request, 'This document is already being processed.')
            return redirect('council_app:pdf_detail', pk=pk)

        # Reset status for reprocessing
        document.status = PDFDocument.Status.PENDING
        document.error_message = ''
        document.processed_pages = 0
        document.save(update_fields=['status', 'error_message', 'processed_pages'])

        # Reset page statuses
        document.pages.all().update(
            status=PDFPage.Status.PENDING,
            error_message='',
            extracted_markdown='',
            processing_time=None,
        )

        # Queue the background task
        from django_q.tasks import async_task
        async_task(
            'council_app.tasks.process_pdf_document',
            document.id,
            task_name=f'pdf_process_{document.id}'
        )

        messages.success(
            request,
            f'Processing started for "{document.title}" ({document.total_pages} pages). '
            f'This may take a while on Raspberry Pi.'
        )
        return redirect('council_app:pdf_detail', pk=pk)


class PDFPageDetailView(DetailView):
    """View a single extracted page's markdown"""
    model = PDFPage
    template_name = 'council_app/churn/pdf_page_detail.html'
    context_object_name = 'page'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        page = self.object
        document = page.document
        context['document'] = document

        # Navigation: prev/next pages
        all_pages = list(document.pages.order_by('page_number'))
        current_idx = next(
            (i for i, p in enumerate(all_pages) if p.pk == page.pk), 0
        )
        context['prev_page'] = all_pages[current_idx - 1] if current_idx > 0 else None
        context['next_page'] = (
            all_pages[current_idx + 1]
            if current_idx < len(all_pages) - 1
            else None
        )
        return context


class PDFDeleteView(View):
    """Delete a PDF document and its ChromaDB collection"""

    def post(self, request, pk):
        document = get_object_or_404(PDFDocument, pk=pk)
        title = document.title

        # Clean up ChromaDB collection
        if document.chromadb_collection:
            try:
                from . import vector_store
                vector_store.delete_collection(document.chromadb_collection)
            except Exception:
                pass  # Non-critical; collection may not exist

        # Delete the file and DB record
        document.file.delete(save=False)
        document.delete()

        messages.success(request, f'PDF "{title}" deleted.')
        return redirect('council_app:pdf_list')


def pdf_status(request, pk):
    """HTMX endpoint for polling PDF processing status"""
    document = get_object_or_404(PDFDocument, pk=pk)
    context = {
        'document': document,
        'pages': document.pages.all().order_by('page_number'),
        'complete_pages': document.pages.filter(status=PDFPage.Status.COMPLETE).count(),
        'error_pages': document.pages.filter(status=PDFPage.Status.ERROR).count(),
    }
    return render(request, 'council_app/churn/partials/pdf_status.html', context)


