from django.urls import path
from . import views

app_name = 'council_app'

urlpatterns = [
    # Main pages
    path('', views.SubmitView.as_view(), name='submit'),
    path('results/<int:pk>/', views.ResultsView.as_view(), name='results'),
    path('history/', views.HistoryView.as_view(), name='history'),
    path('analytics/', views.AnalyticsView.as_view(), name='analytics'),
    path('models/', views.ModelsView.as_view(), name='models'),
    
    # HTMX endpoints
    path('status/<int:pk>/', views.query_status, name='status'),
    
    # Server-Sent Events for real-time updates
    path('events/<int:pk>/stream/', views.query_events_stream, name='events_stream'),
    path('events/<int:pk>/json/', views.query_events_json, name='events_json'),
    
    # Model management
    path('models/add/', views.add_model, name='add_model'),
    path('models/<int:pk>/toggle/', views.toggle_model, name='toggle_model'),
    path('models/<int:pk>/delete/', views.delete_model, name='delete_model'),
    path('models/<int:pk>/timeout/', views.update_model_timeout, name='update_model_timeout'),
    
    # ==========================================================================
    # CHURN MACHINE URLs
    # ==========================================================================
    
    # Project management
    path('churn/', views.ChurnProjectListView.as_view(), name='churn_list'),
    path('churn/settings/', views.ChurnSettingsView.as_view(), name='churn_settings'),
    path('churn/new/', views.CreateProjectView.as_view(), name='churn_create'),
    path('churn/<int:pk>/', views.ProjectDetailView.as_view(), name='churn_project'),
    path('churn/<int:pk>/submit/', views.SubmitIterationView.as_view(), name='churn_submit'),
    
    # Iteration views
    path('churn/iteration/<int:pk>/', views.IterationDetailView.as_view(), name='churn_iteration'),
    path('churn/iteration/<int:pk>/churn/', views.TriggerChurnView.as_view(), name='trigger_churn'),
    path('churn/iteration/<int:pk>/retry/', views.RetryIterationView.as_view(), name='retry_iteration'),
    path('churn/iteration/<int:pk>/stop/', views.StopChurnView.as_view(), name='stop_churn'),
    path('churn/iteration/<int:pk>/branch/', views.BranchView.as_view(), name='churn_branch'),
    path('churn/iteration/<int:pk>/batch/', views.BatchChurnView.as_view(), name='batch_churn'),
    
    # Comparison
    path('churn/compare/<int:pk1>/<int:pk2>/', views.CompareView.as_view(), name='churn_compare'),
    
    # HTMX endpoints for churn
    path('churn/iteration/<int:pk>/status/', views.iteration_status, name='iteration_status'),
    
    # ==========================================================================
    # TECHNICAL REPORT REVIEWER URLs
    # ==========================================================================
    
    # Knowledge base management
    path('churn/kb/', views.KnowledgeBaseListView.as_view(), name='kb_list'),
    path('churn/kb/new/', views.KnowledgeBaseCreateView.as_view(), name='kb_create'),
    path('churn/kb/<int:pk>/', views.KnowledgeBaseDetailView.as_view(), name='kb_detail'),
    
    # Report setup and detail
    path('churn/<int:pk>/report/setup/', views.ReportSetupView.as_view(), name='report_setup'),
    path('churn/<int:pk>/report/', views.ReportDetailView.as_view(), name='report_detail'),
    path('churn/<int:pk>/report/churn-all/', views.TriggerFullReportChurnView.as_view(), name='report_churn_all'),
    
    # Section views
    path('churn/section/<int:pk>/', views.SectionDetailView.as_view(), name='section_detail'),
    path('churn/section/<int:pk>/churn/', views.TriggerSectionChurnView.as_view(), name='section_churn'),
    path('churn/section/<int:pk>/stop/', views.StopSectionChurnView.as_view(), name='stop_section_churn'),
    path('churn/section/<int:pk>/approve/', views.ApproveSectionView.as_view(), name='section_approve'),
    
    # HTMX endpoints for report sections
    path('churn/section/<int:pk>/status/', views.section_status, name='section_status'),

    # ==========================================================================
    # PDF TO MARKDOWN URLs
    # ==========================================================================

    path('churn/pdf/', views.PDFListView.as_view(), name='pdf_list'),
    path('churn/pdf/upload/', views.PDFUploadView.as_view(), name='pdf_upload'),
    path('churn/pdf/<int:pk>/', views.PDFDetailView.as_view(), name='pdf_detail'),
    path('churn/pdf/<int:pk>/process/', views.PDFProcessView.as_view(), name='pdf_process'),
    path('churn/pdf/<int:pk>/delete/', views.PDFDeleteView.as_view(), name='pdf_delete'),
    path('churn/pdf/page/<int:pk>/', views.PDFPageDetailView.as_view(), name='pdf_page_detail'),

    # HTMX endpoints for PDF processing
    path('churn/pdf/<int:pk>/status/', views.pdf_status, name='pdf_status'),
]
