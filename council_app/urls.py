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
    
    # Model management
    path('models/add/', views.add_model, name='add_model'),
    path('models/<int:pk>/toggle/', views.toggle_model, name='toggle_model'),
    path('models/<int:pk>/delete/', views.delete_model, name='delete_model'),
    
    # ==========================================================================
    # CHURN MACHINE URLs
    # ==========================================================================
    
    # Project management
    path('churn/', views.ChurnProjectListView.as_view(), name='churn_list'),
    path('churn/new/', views.CreateProjectView.as_view(), name='churn_create'),
    path('churn/<int:pk>/', views.ProjectDetailView.as_view(), name='churn_project'),
    path('churn/<int:pk>/submit/', views.SubmitIterationView.as_view(), name='churn_submit'),
    
    # Iteration views
    path('churn/iteration/<int:pk>/', views.IterationDetailView.as_view(), name='churn_iteration'),
    path('churn/iteration/<int:pk>/churn/', views.TriggerChurnView.as_view(), name='trigger_churn'),
    path('churn/iteration/<int:pk>/branch/', views.BranchView.as_view(), name='churn_branch'),
    path('churn/iteration/<int:pk>/batch/', views.BatchChurnView.as_view(), name='batch_churn'),
    
    # Comparison
    path('churn/compare/<int:pk1>/<int:pk2>/', views.CompareView.as_view(), name='churn_compare'),
    
    # HTMX endpoints for churn
    path('churn/iteration/<int:pk>/status/', views.iteration_status, name='iteration_status'),
]
