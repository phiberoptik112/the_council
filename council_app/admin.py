from django.contrib import admin
from .models import (
    ModelConfig, Query, Response, Vote, QueryTag,
    CreativeProject, ChurnIteration, IterationFeedback,
    ReportKnowledgeBase, ReportOutline, ReportSection
)


@admin.register(ModelConfig)
class ModelConfigAdmin(admin.ModelAdmin):
    list_display = ['name', 'display_name', 'is_active', 'total_queries', 'total_wins', 'win_rate', 'avg_response_time']
    list_filter = ['is_active']
    search_fields = ['name', 'display_name']
    list_editable = ['is_active']


@admin.register(Query)
class QueryAdmin(admin.ModelAdmin):
    list_display = ['id', 'prompt_preview', 'status', 'council_mode', 'winner_model', 'created_at', 'duration']
    list_filter = ['status', 'council_mode', 'created_at']
    search_fields = ['prompt']
    date_hierarchy = 'created_at'
    
    def prompt_preview(self, obj):
        return obj.prompt[:50] + '...' if len(obj.prompt) > 50 else obj.prompt
    prompt_preview.short_description = 'Prompt'
    
    def winner_model(self, obj):
        if obj.winner:
            return obj.winner.model.name
        return '-'
    winner_model.short_description = 'Winner'


@admin.register(Response)
class ResponseAdmin(admin.ModelAdmin):
    list_display = ['id', 'query', 'model', 'label', 'score', 'is_winner', 'response_time']
    list_filter = ['is_winner', 'model']
    search_fields = ['content']


@admin.register(Vote)
class VoteAdmin(admin.ModelAdmin):
    list_display = ['id', 'query', 'voter_model', 'ranking', 'created_at']
    list_filter = ['voter_model']


@admin.register(QueryTag)
class QueryTagAdmin(admin.ModelAdmin):
    list_display = ['name', 'query_count']
    
    def query_count(self, obj):
        return obj.queries.count()
    query_count.short_description = 'Queries'


# =============================================================================
# CHURN MACHINE ADMIN
# =============================================================================

@admin.register(CreativeProject)
class CreativeProjectAdmin(admin.ModelAdmin):
    list_display = ['title', 'content_type', 'processing_mode', 'iteration_count', 'created_at', 'updated_at']
    list_filter = ['content_type', 'processing_mode', 'default_churn_type']
    search_fields = ['title', 'description']
    date_hierarchy = 'created_at'
    filter_horizontal = ['default_models']


@admin.register(ChurnIteration)
class ChurnIterationAdmin(admin.ModelAdmin):
    list_display = ['id', 'project', 'iteration_number', 'branch_name', 'churn_type', 'status', 'created_at']
    list_filter = ['status', 'churn_type', 'project']
    search_fields = ['content', 'branch_name']
    date_hierarchy = 'created_at'
    raw_id_fields = ['project', 'parent']
    filter_horizontal = ['models_used']


@admin.register(IterationFeedback)
class IterationFeedbackAdmin(admin.ModelAdmin):
    list_display = ['id', 'iteration', 'response_count', 'created_at']
    raw_id_fields = ['iteration']
    search_fields = ['synthesized_feedback']


# =============================================================================
# TECHNICAL REPORT REVIEWER ADMIN
# =============================================================================

@admin.register(ReportKnowledgeBase)
class ReportKnowledgeBaseAdmin(admin.ModelAdmin):
    list_display = ['name', 'description_preview', 'created_at', 'updated_at']
    search_fields = ['name', 'description', 'content']
    date_hierarchy = 'created_at'
    
    def description_preview(self, obj):
        return obj.description[:80] + '...' if len(obj.description) > 80 else obj.description
    description_preview.short_description = 'Description'


@admin.register(ReportOutline)
class ReportOutlineAdmin(admin.ModelAdmin):
    list_display = ['project', 'report_type', 'processing_mode', 'section_count', 'created_at']
    list_filter = ['report_type', 'processing_mode']
    raw_id_fields = ['project', 'knowledgebase']
    search_fields = ['project__title', 'raw_outline']
    date_hierarchy = 'created_at'


@admin.register(ReportSection)
class ReportSectionAdmin(admin.ModelAdmin):
    list_display = ['section_title', 'report_project', 'order', 'status', 'compliance_percent', 'iteration_count', 'updated_at']
    list_filter = ['status', 'report_outline__project']
    search_fields = ['section_title', 'original_content', 'current_content']
    raw_id_fields = ['report_outline']
    date_hierarchy = 'created_at'
    
    def report_project(self, obj):
        return obj.report_outline.project.title
    report_project.short_description = 'Project'
