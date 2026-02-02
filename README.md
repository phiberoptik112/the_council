# The Council

A Django-based web application that orchestrates multiple Large Language Models (LLMs) to collaboratively answer questions through a voting system. The Council queries multiple models in parallel, collects their responses, and uses the models themselves to vote on which response is best.

## Overview

The Council is a multi-model LLM orchestration system that leverages the collective intelligence of multiple language models. Instead of relying on a single model's response, The Council:

1. **Queries multiple models in parallel** - Sends your question to several LLM models simultaneously
2. **Collects responses** - Gathers all responses from the participating models
3. **Models vote** - Each model evaluates and ranks the responses from other models
4. **Determines winner** - Uses voting algorithms (Borda count, majority, etc.) to select the best response

Additionally, The Council includes a **Creative Writing Churn Machine** - an iterative creative writing system that uses the council to provide synthesized feedback, diffs, and expanded content for creative writing projects.

## Key Features

### Core Council System

- **Multi-Model Voting**: Query multiple LLM models simultaneously and have them vote on each other's responses
- **Multiple Voting Modes**:
  - **Ranked Voting** (Borda count): Models rank all responses, points assigned by position
  - **Majority Vote**: Simple majority selection
  - **Synthesis Mode**: Combines best elements from multiple responses into a new response
  - **Debate Mode**: Models critique and refine the winning response
- **Self-Vote Prevention**: Models cannot vote on their own responses (configurable)
- **Async Processing**: Parallel queries for fast response times
- **Robust Error Handling**: Retry logic, timeout handling, and graceful degradation

### Creative Writing Churn Machine

- **Iterative Writing Process**: Submit creative writing and get iterative feedback
- **Three Churn Modes**:
  - **Edit & Refine**: Line-editing with detailed feedback and suggestions
  - **Expand Outline**: Transform outlines into fully developed prose
  - **Explore Variations**: Generate alternative directions and variations
- **Branching Support**: Create branches from any iteration to explore different paths
- **Diff Visualization**: View unified diffs showing changes between iterations
- **Synthesized Feedback**: Multiple models provide feedback that's synthesized into actionable suggestions
- **Batch Processing**: Automatically run multiple iterations

### Web Interface

- **Query Submission**: Submit questions and select models to participate
- **Real-Time Status**: HTMX-powered live updates on query/iteration status
- **Results Dashboard**: View all responses, votes, scores, and the winning response
- **Query History**: Browse and search past queries
- **Analytics**: Model performance statistics, win rates, response times
- **Model Management**: Add, enable/disable, and configure LLM models
- **Project Management**: Create and manage creative writing projects with iteration trees

## Architecture

### Components

1. **`council.py`** - Core LLM council orchestration engine
   - `LLMCouncil` class: Main council orchestrator
   - Async parallel model queries
   - Voting system with multiple modes
   - Response parsing and ranking

2. **`council_app/`** - Django application
   - **Models**: Database models for queries, responses, votes, projects, iterations
   - **Views**: Web interface handlers
   - **Tasks**: Background task processing (Django-Q2)
   - **Forms**: User input forms
   - **Templates**: HTML templates with HTMX integration

3. **`council_app/churn.py`** - Creative writing churn engine
   - `ChurnEngine` class: Iterative writing processor
   - Prompt templates for different churn types
   - Diff generation (unified, HTML, word-level)
   - Response parsing and synthesis

4. **`council_app/tasks.py`** - Background tasks
   - `run_council_query`: Process council queries asynchronously
   - `run_churn_iteration`: Process creative writing iterations
   - `run_batch_churn`: Batch process multiple iterations

### Database Models

- **`ModelConfig`**: LLM model configuration and statistics
- **`Query`**: User-submitted queries with status tracking
- **`Response`**: Individual model responses to queries
- **`Vote`**: Votes cast by models on responses
- **`CreativeProject`**: Creative writing projects
- **`ChurnIteration`**: Individual iterations within a project (supports branching)
- **`IterationFeedback`**: Synthesized feedback from the council

## Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- At least 2 LLM models pulled in Ollama (e.g., `ollama pull phi3:mini`)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd the_council
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database**:
   ```bash
   python manage.py migrate
   ```

5. **Create a superuser** (optional, for admin access):
   ```bash
   python manage.py createsuperuser
   ```

6. **Ensure Ollama is running**:
   ```bash
   ollama serve
   ```

7. **Pull some models** (if not already done):
   ```bash
   ollama pull phi3:mini
   ollama pull tinyllama
   # Add more models as needed
   ```

8. **Start the application**:
   ```bash
   # Option 1: Use the init script (recommended)
   ./init.sh
   
   # Option 2: Manual start
   # Terminal 1: Start Django-Q worker
   python manage.py qcluster
   
   # Terminal 2: Start Django server
   python manage.py runserver
   ```

9. **Access the application**:
   - Open http://localhost:8000 in your browser
   - Admin panel: http://localhost:8000/admin

## Usage

### Council Queries

1. **Submit a Query**:
   - Go to the home page
   - Enter your question/prompt
   - Select at least 2 models from the available models
   - Choose a voting mode (Ranked, Majority, Synthesis, or Debate)
   - Submit

2. **View Results**:
   - Results page shows all responses with scores
   - See individual votes from each model
   - View the winning response highlighted

3. **Browse History**:
   - Access past queries from the History page
   - Search and filter by status

### Creative Writing Churn Machine

1. **Create a Project**:
   - Go to Churn → New Project
   - Enter title, description, content type (Page, Outline, Story, Essay, Poetry)
   - Select default models
   - Create project

2. **Submit Initial Content**:
   - Submit your first draft or outline
   - Select models and churn type (Edit, Expand, or Explore)
   - The council will process and provide feedback

3. **Iterate**:
   - View feedback and suggested revisions
   - Create new iterations from any existing iteration
   - Branch to explore different directions
   - Compare iterations side-by-side

4. **Batch Processing**:
   - Run multiple iterations automatically
   - Option to auto-apply suggestions between iterations

### Model Management

1. **Add Models**:
   - Go to Models page
   - Enter model name (as it appears in Ollama, e.g., `phi3:mini`)
   - Model will be checked for availability

2. **Configure Models**:
   - Enable/disable models
   - View statistics (win rate, response times, total queries)
   - Set model weights for weighted voting (future feature)

## Configuration

### Django Settings

Edit `council_project/settings.py` to configure:

```python
COUNCIL_CONFIG = {
    'OLLAMA_URL': 'http://localhost:11434',
    'DEFAULT_TIMEOUT': 60,
    'DEFAULT_RETRIES': 2,
    'EXCLUDE_SELF_VOTE': True,
}
```

### Django-Q2 Configuration

The queue system is configured in `settings.py`:

```python
Q_CLUSTER = {
    'name': 'council',
    'workers': 2,
    'recycle': 500,
    'timeout': 300,
    'retry': 360,
    'compress': True,
    'save_limit': 250,
    'queue_limit': 500,
    'cpu_affinity': 1,
    'label': 'Council Queue',
    'orm': 'default',
}
```

## Project Structure

```
the_council/
├── council.py                 # Core council orchestration engine
├── manage.py                  # Django management script
├── requirements.txt           # Python dependencies
├── init.sh                    # Startup script
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
│
├── council_project/           # Django project settings
│   ├── settings.py           # Django configuration
│   ├── urls.py               # Root URL configuration
│   ├── wsgi.py               # WSGI application
│   └── asgi.py               # ASGI application
│
├── council_app/              # Main Django application
│   ├── models.py            # Database models
│   ├── views.py             # View handlers
│   ├── urls.py              # URL routing
│   ├── forms.py             # Form definitions
│   ├── admin.py             # Admin configuration
│   ├── tasks.py             # Background tasks
│   ├── churn.py             # Creative writing churn engine
│   │
│   ├── templates/           # HTML templates
│   │   └── council_app/
│   │       ├── submit.html
│   │       ├── results.html
│   │       ├── history.html
│   │       ├── analytics.html
│   │       ├── models.html
│   │       └── churn/       # Churn machine templates
│   │
│   ├── static/              # Static files (CSS, JS)
│   │
│   ├── migrations/          # Database migrations
│   │
│   └── management/          # Custom management commands
│       └── commands/
│           └── seed_models.py
│
└── venv/                    # Virtual environment (not in git)
```

## Technologies

- **Django 5.2+**: Web framework
- **Django-Q2**: Background task queue
- **django-htmx**: HTMX integration for dynamic updates
- **aiohttp**: Async HTTP client for Ollama API
- **Ollama**: Local LLM inference server
- **SQLite**: Default database (can be switched to PostgreSQL/MySQL)

## Development

### Running Tests

```bash
python manage.py test
```

### Creating Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### Accessing Admin Panel

1. Create superuser: `python manage.py createsuperuser`
2. Access at: http://localhost:8000/admin

### Logging

Logs are written to:
- Console (with verbose formatting)
- `council_debug.log` file

Log levels can be configured in `settings.py` under `LOGGING`.

## Troubleshooting

### Ollama Connection Issues

- Ensure Ollama is running: `ollama serve`
- Check Ollama URL in settings matches your setup
- Verify models are pulled: `ollama list`

### Django-Q Worker Not Starting

- Check if port is already in use
- Review logs for errors
- Ensure database migrations are up to date

### Models Not Responding

- Verify models are available in Ollama
- Check model names match exactly (including tags, e.g., `phi3:mini`)
- Increase timeout in settings if models are slow
- Check Ollama logs for errors

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

Built with Django, Ollama, and the power of multiple LLMs working together.
