from django.db import models
from django.utils import timezone


class ModelConfig(models.Model):
    """Configuration for an LLM model in the council"""
    name = models.CharField(max_length=100, unique=True)  # e.g., "phi3:mini"
    display_name = models.CharField(max_length=100, blank=True)  # Friendly name
    is_active = models.BooleanField(default=True)
    weight = models.FloatField(default=1.0)  # For weighted voting
    category_weights = models.JSONField(default=dict, blank=True)  # {"coding": 1.5, "creative": 0.8}
    timeout = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Timeout in seconds (uses global default if blank)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Statistics (updated after each query)
    total_queries = models.IntegerField(default=0)
    total_wins = models.IntegerField(default=0)
    avg_response_time = models.FloatField(default=0.0)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return self.display_name or self.name
    
    @property
    def win_rate(self):
        if self.total_queries == 0:
            return 0
        return round((self.total_wins / self.total_queries) * 100, 1)
    
    def update_stats(self, won: bool, response_time: float):
        """Update model statistics after a query"""
        self.total_queries += 1
        if won:
            self.total_wins += 1
        # Running average for response time
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time * 0.9) + (response_time * 0.1)
        self.save()


class Query(models.Model):
    """A query submitted to the council"""
    
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        GATHERING = 'gathering', 'Gathering Responses'
        VOTING = 'voting', 'Voting'
        COMPLETE = 'complete', 'Complete'
        ERROR = 'error', 'Error'
    
    class Mode(models.TextChoices):
        RANKED = 'ranked', 'Ranked Voting'
        MAJORITY = 'majority', 'Majority Vote'
        SYNTHESIS = 'synthesis', 'Synthesis'
        DEBATE = 'debate', 'Debate'
    
    prompt = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )
    council_mode = models.CharField(
        max_length=20,
        choices=Mode.choices,
        default=Mode.RANKED
    )
    error_message = models.TextField(blank=True)
    
    # Stores the winning response ID for quick access
    winner = models.ForeignKey(
        'Response',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='won_queries'
    )
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Queries'
    
    def __str__(self):
        return f"Query #{self.id}: {self.prompt[:50]}..."
    
    @property
    def duration(self):
        if self.completed_at and self.created_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None
    
    def mark_complete(self, winner_response=None):
        self.status = self.Status.COMPLETE
        self.completed_at = timezone.now()
        if winner_response:
            self.winner = winner_response
        self.save()
    
    def mark_error(self, message: str):
        self.status = self.Status.ERROR
        self.error_message = message
        self.completed_at = timezone.now()
        self.save()


class Response(models.Model):
    """A response from a single model to a query"""
    query = models.ForeignKey(Query, on_delete=models.CASCADE, related_name='responses')
    model = models.ForeignKey(ModelConfig, on_delete=models.CASCADE, related_name='responses')
    content = models.TextField()
    label = models.CharField(max_length=1)  # A, B, C, etc.
    response_time = models.FloatField(default=0.0)  # in seconds
    is_winner = models.BooleanField(default=False)
    score = models.IntegerField(default=0)
    is_validated = models.BooleanField(default=True)  # Whether response passed quality validation
    validation_notes = models.TextField(blank=True, default='')  # Reasons for validation failure
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['label']
        unique_together = [['query', 'label']]
    
    def __str__(self):
        return f"Response {self.label} from {self.model.name}"


class Vote(models.Model):
    """A vote cast by a model on a query's responses"""
    query = models.ForeignKey(Query, on_delete=models.CASCADE, related_name='votes')
    voter_model = models.ForeignKey(ModelConfig, on_delete=models.CASCADE, related_name='votes_cast')
    ranking = models.JSONField()  # ["A", "B", "C"]
    raw_response = models.TextField(blank=True)  # The actual vote response for debugging
    voted_on_models = models.JSONField(default=list)  # Models that were voted on
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = [['query', 'voter_model']]
    
    def __str__(self):
        return f"Vote by {self.voter_model.name} on Query #{self.query.id}"


class QueryTag(models.Model):
    """Tags/categories for queries (for organization and analytics)"""
    name = models.CharField(max_length=50, unique=True)
    queries = models.ManyToManyField(Query, related_name='tags', blank=True)
    
    def __str__(self):
        return self.name


class QueryEvent(models.Model):
    """
    Tracks granular progress events during query processing.
    Used for real-time UI feedback and debugging.
    """
    
    class EventType(models.TextChoices):
        # Query lifecycle
        QUERY_STARTED = 'started', 'Query Started'
        
        # Model response gathering
        MODEL_QUERYING = 'querying', 'Querying Model'
        MODEL_SUCCESS = 'success', 'Model Responded'
        MODEL_ERROR = 'error', 'Model Error'
        MODEL_TIMEOUT = 'timeout', 'Model Timeout'
        MODEL_RETRY = 'retry', 'Retrying Model'
        
        # Voting phase
        VOTING_STARTED = 'voting_start', 'Voting Started'
        MODEL_VOTING = 'voting', 'Model Voting'
        MODEL_VOTED = 'voted', 'Model Voted'
        
        # Completion
        QUERY_COMPLETE = 'complete', 'Query Complete'
        QUERY_ERROR = 'query_error', 'Query Error'
    
    query = models.ForeignKey(
        Query,
        on_delete=models.CASCADE,
        related_name='events'
    )
    event_type = models.CharField(
        max_length=20,
        choices=EventType.choices
    )
    model_name = models.CharField(max_length=100, blank=True)
    message = models.TextField(blank=True)
    raw_data = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        if self.model_name:
            return f"{self.get_event_type_display()} - {self.model_name}"
        return self.get_event_type_display()


# =============================================================================
# CHURN MACHINE MODELS - Creative Writing Iteration System
# =============================================================================

class CreativeProject(models.Model):
    """A creative writing project that contains multiple iterations"""
    
    class ContentType(models.TextChoices):
        PAGE = 'page', 'Page of Writing'
        OUTLINE = 'outline', 'Outline'
        STORY = 'story', 'Story'
        ESSAY = 'essay', 'Essay'
        POETRY = 'poetry', 'Poetry'
        REPORT = 'report', 'Technical Report'
    
    class ProcessingMode(models.TextChoices):
        MANUAL = 'manual', 'Manual (trigger each iteration)'
        AUTO_BATCH = 'auto_batch', 'Automatic Batch'
    
    class ChurnType(models.TextChoices):
        EDIT = 'edit', 'Edit & Refine'
        EXPAND = 'expand', 'Expand Outline'
        EXPLORE = 'explore', 'Explore Variations'
        REPORT = 'report', 'Report Review'
    
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    content_type = models.CharField(
        max_length=20,
        choices=ContentType.choices,
        default=ContentType.PAGE
    )
    processing_mode = models.CharField(
        max_length=20,
        choices=ProcessingMode.choices,
        default=ProcessingMode.MANUAL
    )
    default_churn_type = models.CharField(
        max_length=20,
        choices=ChurnType.choices,
        default=ChurnType.EDIT
    )
    default_models = models.ManyToManyField(
        ModelConfig,
        related_name='churn_projects',
        blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return self.title
    
    @property
    def root_iteration(self):
        """Get the root iteration (original submission)"""
        return self.iterations.filter(parent__isnull=True).first()
    
    @property
    def iteration_count(self):
        """Total number of iterations in this project"""
        return self.iterations.count()
    
    @property
    def latest_iteration(self):
        """Get the most recent iteration"""
        return self.iterations.order_by('-created_at').first()
    
    def get_iteration_tree(self):
        """Build a tree structure of all iterations"""
        root = self.root_iteration
        if not root:
            return None
        return root.get_subtree()


class ChurnIteration(models.Model):
    """An iteration within a creative project - supports branching"""
    
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        COMPLETE = 'complete', 'Complete'
        ARCHIVED = 'archived', 'Archived'
        ERROR = 'error', 'Error'
    
    project = models.ForeignKey(
        CreativeProject,
        on_delete=models.CASCADE,
        related_name='iterations'
    )
    parent = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='children'
    )
    branch_name = models.CharField(max_length=100, blank=True)
    iteration_number = models.PositiveIntegerField(default=1)
    content = models.TextField()
    content_diff = models.TextField(blank=True)  # Unified diff from parent
    churn_type = models.CharField(
        max_length=20,
        choices=CreativeProject.ChurnType.choices,
        default=CreativeProject.ChurnType.EDIT
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )
    error_message = models.TextField(blank=True)
    debug_info = models.JSONField(default=dict, blank=True, help_text="Debug information including model responses and validation results")
    models_used = models.ManyToManyField(
        ModelConfig,
        related_name='churn_iterations',
        blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    cancel_requested = models.BooleanField(
        default=False,
        help_text="When True, the running task will stop at the next stage boundary"
    )
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        branch = f" ({self.branch_name})" if self.branch_name else ""
        return f"{self.project.title} - Iteration {self.iteration_number}{branch}"
    
    @property
    def is_root(self):
        """Check if this is the root/original iteration"""
        return self.parent is None
    
    @property
    def depth(self):
        """Get the depth in the tree (0 for root)"""
        depth = 0
        current = self
        while current.parent:
            depth += 1
            current = current.parent
        return depth
    
    def get_branch_path(self):
        """Returns list of ancestors from root to self"""
        path = []
        current = self
        while current:
            path.insert(0, current)
            current = current.parent
        return path
    
    def get_subtree(self):
        """Returns this iteration and all descendants as a nested dict"""
        return {
            'iteration': self,
            'children': [child.get_subtree() for child in self.children.all()]
        }
    
    def get_siblings(self):
        """Get other iterations that share the same parent"""
        if self.parent:
            return self.parent.children.exclude(pk=self.pk)
        return ChurnIteration.objects.filter(
            project=self.project,
            parent__isnull=True
        ).exclude(pk=self.pk)
    
    def mark_processing(self):
        self.status = self.Status.PROCESSING
        self.save()
    
    def mark_complete(self):
        self.status = self.Status.COMPLETE
        self.completed_at = timezone.now()
        self.save()
    
    def mark_error(self, message: str):
        self.status = self.Status.ERROR
        self.error_message = message
        self.completed_at = timezone.now()
        self.save()


class IterationFeedback(models.Model):
    """Synthesized feedback from the council for an iteration"""
    
    iteration = models.OneToOneField(
        ChurnIteration,
        on_delete=models.CASCADE,
        related_name='feedback'
    )
    synthesized_feedback = models.TextField()
    suggested_content = models.TextField(blank=True)  # Council's suggested revision
    raw_responses = models.JSONField(default=list)  # Individual model responses
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Feedback for {self.iteration}"
    
    @property
    def response_count(self):
        """Number of individual responses that were synthesized"""
        return len(self.raw_responses)


class ChurnConfig(models.Model):
    """User-configurable performance settings for the Churn Machine (singleton)"""
    # Ollama options
    num_predict = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Max tokens to generate (blank=unlimited)"
    )
    keep_alive = models.CharField(
        max_length=20,
        blank=True,
        help_text="e.g. 5m, 0, blank=default"
    )
    num_ctx = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Context window (blank=model default)"
    )
    # Churn optimizations
    sequential_models = models.BooleanField(
        default=True,
        help_text="Run models one at a time (better for Pi)"
    )
    max_content_chars = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Max chars of content in prompts (blank=no limit)"
    )
    max_synthesis_response_chars = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Max chars per response in synthesis"
    )
    use_streaming = models.BooleanField(
        default=False,
        help_text="Stream Ollama response to avoid timeout"
    )
    debug_full_responses = models.BooleanField(
        default=False,
        help_text="Store full model responses in debug_info (can be large)"
    )
    debug_response_max_chars = models.PositiveIntegerField(
        default=500,
        help_text="Max chars per response in debug_info when debug_full_responses is False"
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Churn Configuration"
        verbose_name_plural = "Churn Configuration"

    @classmethod
    def get_instance(cls):
        """Get or create the singleton config"""
        obj, _ = cls.objects.get_or_create(pk=1, defaults={})
        return obj

    @classmethod
    def get_effective_config(cls):
        """Merge DB config with COUNCIL_CONFIG. DB values override when set."""
        from django.conf import settings as django_settings
        db = cls.get_instance()
        cfg = dict(getattr(django_settings, 'COUNCIL_CONFIG', {}))
        if db.num_predict is not None:
            cfg['NUM_PREDICT'] = db.num_predict
        if db.keep_alive:
            cfg['KEEP_ALIVE'] = db.keep_alive
        if db.num_ctx is not None:
            cfg['NUM_CTX'] = db.num_ctx
        cfg['CHURN_SEQUENTIAL_MODELS'] = db.sequential_models
        if db.max_content_chars is not None:
            cfg['CHURN_MAX_CONTENT_CHARS'] = db.max_content_chars
        if db.max_synthesis_response_chars is not None:
            cfg['CHURN_MAX_SYNTHESIS_RESPONSE_CHARS'] = db.max_synthesis_response_chars
        cfg['CHURN_USE_STREAMING'] = db.use_streaming
        cfg['CHURN_DEBUG_FULL_RESPONSES'] = db.debug_full_responses
        cfg['CHURN_DEBUG_RESPONSE_MAX_CHARS'] = db.debug_response_max_chars
        return cfg


# =============================================================================
# TECHNICAL REPORT REVIEWER MODELS
# =============================================================================

class ReportKnowledgeBase(models.Model):
    """
    A collection of guidelines/templates that inform report structure.
    Could be: style guides, templates, regulatory requirements, etc.
    Reusable across multiple report projects.
    """
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    content = models.TextField(
        help_text="Knowledge content in markdown format (style guides, templates, requirements)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Report Knowledge Base'
        verbose_name_plural = 'Report Knowledge Bases'
    
    def __str__(self):
        return self.name
    
    def get_relevant_context(self, section_title: str, section_content: str) -> str:
        """
        Extract relevant portions of the knowledgebase for a given section.
        V1: Returns full content.
        V2: Could use keyword matching or semantic search via ChromaDB.
        """
        return self.content


class ReportOutline(models.Model):
    """
    Structured outline for a technical report.
    Attaches report-specific data to a CreativeProject via OneToOneField.
    """
    
    class ReportType(models.TextChoices):
        TECHNICAL = 'technical', 'Technical Report'
        RESEARCH = 'research', 'Research Paper'
        PROPOSAL = 'proposal', 'Proposal'
    
    class ProcessingMode(models.TextChoices):
        SEQUENTIAL = 'sequential', 'Sequential (one at a time)'
        PARALLEL = 'parallel', 'Parallel (all at once)'
    
    project = models.OneToOneField(
        CreativeProject,
        on_delete=models.CASCADE,
        related_name='report_outline'
    )
    raw_outline = models.TextField(help_text="Original outline as submitted (markdown)")
    parsed_sections = models.JSONField(
        default=list,
        help_text="Parsed sections: [{id, title, content, level, parent_id, order}]"
    )
    knowledgebase = models.ForeignKey(
        ReportKnowledgeBase,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='report_outlines'
    )
    report_type = models.CharField(
        max_length=20,
        choices=ReportType.choices,
        default=ReportType.TECHNICAL
    )
    target_audience = models.CharField(max_length=200, blank=True)
    processing_mode = models.CharField(
        max_length=20,
        choices=ProcessingMode.choices,
        default=ProcessingMode.SEQUENTIAL
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"Report Outline for {self.project.title}"
    
    def parse_outline(self):
        """Parse raw_outline markdown into structured sections using OutlineParser."""
        from .report_churn import OutlineParser
        parser = OutlineParser()
        self.parsed_sections = parser.parse(self.raw_outline)
        self.save()
        return self.parsed_sections
    
    @property
    def section_count(self):
        return len(self.parsed_sections)
    
    @property
    def approved_section_count(self):
        return self.sections.filter(status=ReportSection.Status.APPROVED).count()
    
    @property
    def progress_percent(self):
        total = self.sections.count()
        if total == 0:
            return 0
        approved = self.sections.filter(status=ReportSection.Status.APPROVED).count()
        return round((approved / total) * 100)


class ReportSection(models.Model):
    """
    Tracks per-section iteration state for a technical report.
    Self-contained with its own feedback/status tracking.
    """
    
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        IN_PROGRESS = 'in_progress', 'In Progress'
        REVIEW = 'review', 'Awaiting Review'
        APPROVED = 'approved', 'Approved'
        NEEDS_REVISION = 'needs_revision', 'Needs Revision'
    
    report_outline = models.ForeignKey(
        ReportOutline,
        on_delete=models.CASCADE,
        related_name='sections'
    )
    section_id = models.CharField(
        max_length=50,
        help_text="References parsed_sections[].id"
    )
    section_title = models.CharField(max_length=200)
    order = models.PositiveIntegerField(default=0)
    
    # Content tracking
    original_content = models.TextField(blank=True)
    current_content = models.TextField(blank=True)
    content_diff = models.TextField(blank=True)
    
    # Knowledge context used for this section
    knowledge_context = models.TextField(blank=True)
    
    # Status and progress
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )
    iteration_count = models.PositiveIntegerField(default=0)
    compliance_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Compliance score from 0.0 to 1.0"
    )
    
    # Feedback from the council
    council_feedback = models.JSONField(
        default=dict,
        blank=True,
        help_text="Structured feedback from latest review"
    )
    raw_responses = models.JSONField(
        default=list,
        blank=True,
        help_text="Individual model responses"
    )
    debug_info = models.JSONField(
        default=dict,
        blank=True,
        help_text="Live debug info during processing (model responses, etc.)"
    )
    cancel_requested = models.BooleanField(
        default=False,
        help_text="When True, the running task will stop at the next stage boundary"
    )
    
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['order', 'created_at']
        unique_together = [['report_outline', 'section_id']]
    
    def __str__(self):
        return f"{self.section_title} ({self.get_status_display()})"
    
    @property
    def compliance_percent(self):
        """Compliance score as a percentage for display."""
        if self.compliance_score is None:
            return None
        return round(self.compliance_score * 100)
    
    def mark_in_progress(self):
        self.status = self.Status.IN_PROGRESS
        self.save()
    
    def mark_review(self):
        self.status = self.Status.REVIEW
        self.save()
    
    def mark_approved(self):
        self.status = self.Status.APPROVED
        self.save()
    
    def mark_needs_revision(self):
        self.status = self.Status.NEEDS_REVISION
        self.save()
    
    def mark_error(self, message: str):
        self.status = self.Status.PENDING
        self.error_message = message
        self.save()


# =============================================================================
# PDF TO MARKDOWN PROCESSING MODELS
# =============================================================================

class PDFDocument(models.Model):
    """
    A PDF document uploaded for vision-based markdown extraction.
    Processed page-by-page using a single Ollama vision model.
    """

    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        COMPLETE = 'complete', 'Complete'
        ERROR = 'error', 'Error'

    title = models.CharField(max_length=200)
    file = models.FileField(upload_to='pdfs/')
    total_pages = models.PositiveIntegerField(default=0)
    processed_pages = models.PositiveIntegerField(default=0)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )
    model = models.ForeignKey(
        ModelConfig,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='pdf_documents',
        help_text="Vision model used for image-to-markdown conversion"
    )
    error_message = models.TextField(blank=True)
    chromadb_collection = models.CharField(
        max_length=200,
        blank=True,
        help_text="ChromaDB collection name for vector lookups"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'PDF Document'
        verbose_name_plural = 'PDF Documents'

    def __str__(self):
        return f"{self.title} ({self.processed_pages}/{self.total_pages} pages)"

    @property
    def progress_percent(self):
        if self.total_pages == 0:
            return 0
        return round((self.processed_pages / self.total_pages) * 100)

    def mark_processing(self):
        self.status = self.Status.PROCESSING
        self.save(update_fields=['status'])

    def mark_complete(self):
        self.status = self.Status.COMPLETE
        self.save(update_fields=['status'])

    def mark_error(self, message: str):
        self.status = self.Status.ERROR
        self.error_message = message
        self.save(update_fields=['status', 'error_message'])


class PDFPage(models.Model):
    """
    A single page extracted from a PDF document.
    Stores the vision model's markdown output for that page.
    """

    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        COMPLETE = 'complete', 'Complete'
        ERROR = 'error', 'Error'

    document = models.ForeignKey(
        PDFDocument,
        on_delete=models.CASCADE,
        related_name='pages'
    )
    page_number = models.PositiveIntegerField()
    extracted_markdown = models.TextField(blank=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )
    processing_time = models.FloatField(
        null=True,
        blank=True,
        help_text="Seconds taken to process this page"
    )
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['page_number']
        unique_together = [['document', 'page_number']]
        verbose_name = 'PDF Page'
        verbose_name_plural = 'PDF Pages'

    def __str__(self):
        return f"{self.document.title} - Page {self.page_number}"

    def mark_processing(self):
        self.status = self.Status.PROCESSING
        self.save(update_fields=['status'])

    def mark_complete(self, markdown: str, processing_time: float):
        self.extracted_markdown = markdown
        self.processing_time = processing_time
        self.status = self.Status.COMPLETE
        self.save(update_fields=['extracted_markdown', 'processing_time', 'status'])

    def mark_error(self, message: str):
        self.status = self.Status.ERROR
        self.error_message = message
        self.save(update_fields=['status', 'error_message'])
