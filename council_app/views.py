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
    CreativeProject, ChurnIteration, IterationFeedback
)
from .forms import (
    QueryForm, ModelConfigForm, AddModelForm,
    CreativeProjectForm, SubmitContentForm, BranchForm,
    BatchChurnForm, TriggerChurnForm
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
    """
    def event_stream():
        last_event_id = 0
        heartbeat_interval = 15  # seconds
        last_heartbeat = time.time()
        
        while True:
            try:
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
        context['latest_iteration'] = project.latest_iteration
        
        # Get branch form for quick branching
        context['branch_form'] = BranchForm()
        context['trigger_form'] = TriggerChurnForm()
        context['batch_form'] = BatchChurnForm()
        
        return context


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
        
        # Parse diff for display
        if iteration.content_diff:
            context['diff_lines'] = parse_diff_for_display(iteration.content_diff)
        
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
    return render(request, 'council_app/churn/partials/iteration_status.html', {
        'iteration': iteration
    })


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


