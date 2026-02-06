"""
Background tasks for running council queries.
Uses Django-Q2 for async task execution.
"""
import asyncio
import logging
from django.conf import settings
from django.utils import timezone
from asgiref.sync import sync_to_async

from .models import Query, Response, Vote, ModelConfig, ChurnIteration, IterationFeedback, CreativeProject, QueryEvent

# Set up loggers for debugging
logger = logging.getLogger('council_app.tasks')
council_logger = logging.getLogger('council')


# =============================================================================
# EVENT LOGGING HELPERS
# =============================================================================

async def log_query_event(query_id: int, event_type: str, model_name: str = "", 
                          message: str = "", raw_data: dict = None):
    """
    Log a query event to the database for real-time tracking.
    
    Args:
        query_id: ID of the Query object
        event_type: One of QueryEvent.EventType choices
        model_name: Name of the model (if applicable)
        message: Human-readable message
        raw_data: Additional data to store as JSON
    """
    try:
        await sync_to_async(QueryEvent.objects.create)(
            query_id=query_id,
            event_type=event_type,
            model_name=model_name,
            message=message,
            raw_data=raw_data or {}
        )
    except Exception as e:
        logger.warning(f"Failed to log query event: {e}")


def create_event_callback(query_id: int):
    """
    Create a callback function for the council to report progress events.
    
    The callback will be called by the council's query methods with:
    - model: str - the model name (or None for general events)
    - stage: str - the event stage/type
    - data: Any - additional data (response object, error message, etc.)
    """
    def callback(model: str, stage: str, data):
        """Synchronous wrapper that schedules async event logging."""
        import time
        
        # Map council stages to QueryEvent types
        event_map = {
            'start': QueryEvent.EventType.QUERY_STARTED,
            'gathering': QueryEvent.EventType.QUERY_STARTED,
            'querying': QueryEvent.EventType.MODEL_QUERYING,
            'success': QueryEvent.EventType.MODEL_SUCCESS,
            'error': QueryEvent.EventType.MODEL_ERROR,
            'timeout': QueryEvent.EventType.MODEL_TIMEOUT,
            'retry': QueryEvent.EventType.MODEL_RETRY,
            'voting_start': QueryEvent.EventType.VOTING_STARTED,
            'voting': QueryEvent.EventType.MODEL_VOTING,
            'voted': QueryEvent.EventType.MODEL_VOTED,
            'synthesizing': QueryEvent.EventType.MODEL_QUERYING,
            'debating': QueryEvent.EventType.MODEL_QUERYING,
            'complete': QueryEvent.EventType.QUERY_COMPLETE,
        }
        
        event_type = event_map.get(stage, QueryEvent.EventType.QUERY_STARTED)
        model_name = model or ""
        
        # Build message based on stage
        if stage == 'success' and data:
            message = f"Responded in {data.response_time:.1f}s"
            raw_data = {'response_time': data.response_time, 'response_length': len(data.response)}
        elif stage == 'error':
            message = str(data) if data else "Unknown error"
            raw_data = {'error': message}
        elif stage == 'timeout':
            message = f"Timed out after waiting"
            raw_data = {}
        elif stage == 'voted' and data:
            message = f"Ranked: {' > '.join(data)}"
            raw_data = {'ranking': data}
        elif stage == 'querying':
            message = "Querying model..."
            raw_data = {}
        elif stage == 'voting':
            message = "Casting vote..."
            raw_data = {}
        else:
            message = stage.replace('_', ' ').title()
            raw_data = {}
        
        # Log synchronously (this runs in background task context)
        try:
            QueryEvent.objects.create(
                query_id=query_id,
                event_type=event_type,
                model_name=model_name,
                message=message,
                raw_data=raw_data
            )
        except Exception as e:
            logger.warning(f"Failed to log event: {e}")
    
    return callback


def run_council_query(query_id: int, model_ids: list):
    """
    Background task to run a council query.
    
    Args:
        query_id: ID of the Query object
        model_ids: List of ModelConfig IDs to use
    """
    logger.info(f"=== Starting council query {query_id} ===")
    logger.debug(f"Model IDs: {model_ids}")
    
    try:
        query = Query.objects.get(id=query_id)
        logger.debug(f"Found query: {query.prompt[:50]}...")
    except Query.DoesNotExist:
        logger.error(f"Query {query_id} not found in database")
        return f"Query {query_id} not found"
    
    try:
        # Get model configs
        model_configs = list(ModelConfig.objects.filter(id__in=model_ids, is_active=True))
        logger.info(f"Found {len(model_configs)} active models from {len(model_ids)} requested")
        
        if len(model_configs) < 2:
            logger.error(f"Not enough active models: {len(model_configs)} < 2")
            query.mark_error("Not enough active models selected")
            return "Not enough models"
        
        model_names = [m.name for m in model_configs]
        logger.info(f"Using models: {model_names}")
        
        # Update status to gathering
        query.status = Query.Status.GATHERING
        query.save()
        logger.debug("Query status set to GATHERING")
        
        # Import and run council
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from council import LLMCouncil, CouncilMode
        
        # Get council config from settings
        config = getattr(settings, 'COUNCIL_CONFIG', {})
        ollama_url = config.get('OLLAMA_URL', 'http://localhost:11434')
        default_timeout = config.get('DEFAULT_TIMEOUT', 300)
        retries = config.get('DEFAULT_RETRIES', 2)
        
        # Use per-model timeouts if available, otherwise use default
        model_timeouts = [m.timeout for m in model_configs if m.timeout]
        timeout = max(model_timeouts) if model_timeouts else default_timeout
        
        logger.info(f"Council config - Ollama URL: {ollama_url}, Timeout: {timeout}s, Retries: {retries}")
        
        # Map mode string to enum
        mode_map = {
            'ranked': CouncilMode.RANKED,
            'majority': CouncilMode.MAJORITY,
            'synthesis': CouncilMode.SYNTHESIS,
            'debate': CouncilMode.DEBATE,
        }
        mode = mode_map.get(query.council_mode, CouncilMode.RANKED)
        logger.info(f"Council mode: {mode.value}")
        
        # Create council instance
        logger.debug(f"Creating LLMCouncil with {len(model_names)} models")
        council = LLMCouncil(
            models=model_names,
            ollama_url=ollama_url,
            timeout=timeout,
            retries=retries,
            exclude_self_vote=config.get('EXCLUDE_SELF_VOTE', True)
        )
        
        # Run the council (async)
        logger.info("Starting async council processing")
        result = asyncio.run(run_council_async(council, query, model_configs, mode))
        
        logger.info(f"=== Council query {query_id} completed successfully ===")
        return f"Council query {query_id} completed successfully"
        
    except Exception as e:
        logger.exception(f"Council query {query_id} failed with exception: {e}")
        query.mark_error(str(e))
        return f"Council query {query_id} failed: {e}"


async def run_council_async(council, query, model_configs, mode):
    """
    Async function to run the council and save results.
    """
    from council import CouncilMode
    
    # Create model name to config mapping
    model_map = {m.name: m for m in model_configs}
    
    # Create event callback for progress tracking
    event_callback = create_event_callback(query.id)
    
    # Log query start
    await log_query_event(
        query.id, 
        QueryEvent.EventType.QUERY_STARTED,
        message=f"Starting query with {len(model_configs)} models",
        raw_data={'models': [m.name for m in model_configs]}
    )
    
    # Update status
    query.status = Query.Status.GATHERING
    await sync_to_async(query.save)()
    
    # Get all responses with callback for progress tracking
    responses = await council.get_all_responses_async(query.prompt, callback=event_callback)
    
    if not responses:
        await log_query_event(
            query.id,
            QueryEvent.EventType.QUERY_ERROR,
            message="No responses received from any model"
        )
        await sync_to_async(query.mark_error)("No responses received from any model")
        return None
    
    # Save responses to database
    labels = council.LABELS
    db_responses = []
    
    for i, resp in enumerate(responses):
        if resp.error:
            continue
            
        model_config = model_map.get(resp.model)
        if not model_config:
            continue
        
        db_response = await sync_to_async(Response.objects.create)(
            query=query,
            model=model_config,
            content=resp.response,
            label=labels[i],
            response_time=resp.response_time
        )
        db_responses.append(db_response)
    
    if not db_responses:
        await sync_to_async(query.mark_error)("Failed to save any valid responses")
        return None
    
    # Update status to voting
    query.status = Query.Status.VOTING
    await sync_to_async(query.save)()
    
    # Log voting start
    await log_query_event(
        query.id,
        QueryEvent.EventType.VOTING_STARTED,
        message=f"Starting voting phase with {len(responses)} responses"
    )
    
    # Conduct voting with callback
    vote_results = await council.conduct_vote_async(query.prompt, responses, callback=event_callback)
    
    # Save votes to database
    for vote_data in vote_results.get('all_votes', []):
        voter_model = model_map.get(vote_data['voter'])
        if voter_model:
            await sync_to_async(Vote.objects.create)(
                query=query,
                voter_model=voter_model,
                ranking=vote_data['ranking'],
                raw_response=vote_data.get('raw_response', ''),
                voted_on_models=vote_data.get('voted_on', [])
            )
    
    # Update response scores
    scores = vote_results.get('scores', {})
    winner_label = vote_results.get('winner_label', 'A')
    
    for db_response in db_responses:
        db_response.score = scores.get(db_response.label, 0)
        db_response.is_winner = (db_response.label == winner_label)
        await sync_to_async(db_response.save)()
        
        # Update model statistics
        await sync_to_async(db_response.model.update_stats)(
            won=db_response.is_winner,
            response_time=db_response.response_time
        )
    
    # Find and set winner
    winner_response = next((r for r in db_responses if r.is_winner), None)
    
    # Handle synthesis/debate modes
    if mode == CouncilMode.SYNTHESIS and winner_response:
        synthesis_result = await council.synthesize_response_async(
            query.prompt, responses
        )
        if synthesis_result and not synthesis_result.error:
            winner_response.content = synthesis_result.response
            await sync_to_async(winner_response.save)()
    
    elif mode == CouncilMode.DEBATE and winner_response:
        winning_model_response = next(
            (r for r in responses if r.model == winner_response.model.name), 
            None
        )
        if winning_model_response:
            debate_result = await council.debate_and_refine_async(
                query.prompt, winning_model_response, responses
            )
            if debate_result and not debate_result.error:
                winner_response.content = debate_result.response
                await sync_to_async(winner_response.save)()
    
    # Mark query complete
    await sync_to_async(query.mark_complete)(winner_response)
    
    # Log completion
    await log_query_event(
        query.id,
        QueryEvent.EventType.QUERY_COMPLETE,
        model_name=winner_response.model.name if winner_response else "",
        message=f"Query complete. Winner: {winner_response.model.name if winner_response else 'N/A'}",
        raw_data={'scores': scores, 'winner_label': winner_label}
    )
    
    return vote_results


def check_model_availability(model_name: str) -> dict:
    """
    Task to check if a model is available in Ollama.
    
    Returns:
        dict with 'available' (bool) and 'message' (str)
    """
    import aiohttp
    
    async def check():
        config = getattr(settings, 'COUNCIL_CONFIG', {})
        ollama_url = config.get('OLLAMA_URL', 'http://localhost:11434')
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = [m.get('name', '') for m in data.get('models', [])]
                        available = any(model_name in m or m.startswith(model_name.split(':')[0]) for m in models)
                        return {
                            'available': available,
                            'message': 'Model found' if available else f'Model {model_name} not found in Ollama'
                        }
                    else:
                        return {'available': False, 'message': f'Ollama returned status {resp.status}'}
        except Exception as e:
            return {'available': False, 'message': f'Error connecting to Ollama: {e}'}
    
    return asyncio.run(check())


# =============================================================================
# CHURN MACHINE TASKS
# =============================================================================

def run_churn_iteration(iteration_id: int, model_ids: list, context: str = ""):
    """
    Background task to run a churn iteration.
    
    Args:
        iteration_id: ID of the ChurnIteration object
        model_ids: List of ModelConfig IDs to use
        context: Optional context/genre for the writing
    """
    logger.info(f"=== Starting churn iteration {iteration_id} ===")
    logger.debug(f"Model IDs: {model_ids}, Context: '{context}'")
    
    try:
        iteration = ChurnIteration.objects.get(id=iteration_id)
        logger.debug(f"Found iteration: {iteration}, Project: {iteration.project.title}")
    except ChurnIteration.DoesNotExist:
        logger.error(f"Iteration {iteration_id} not found in database")
        return f"Iteration {iteration_id} not found"
    
    try:
        # Get model configs
        model_configs = list(ModelConfig.objects.filter(id__in=model_ids, is_active=True))
        logger.info(f"Found {len(model_configs)} active models from {len(model_ids)} requested")
        
        if len(model_configs) < 2:
            logger.error(f"Not enough active models: {len(model_configs)} < 2")
            iteration.mark_error("Not enough active models selected")
            return "Not enough models"
        
        model_names = [m.name for m in model_configs]
        logger.info(f"Using models: {model_names}")
        
        # Update status to processing
        iteration.mark_processing()
        logger.debug(f"Iteration status set to PROCESSING")
        
        # Import churn engine
        from .churn import ChurnEngine, ChurnType
        
        # Get council config from settings
        config = getattr(settings, 'COUNCIL_CONFIG', {})
        ollama_url = config.get('OLLAMA_URL', 'http://localhost:11434')
        default_timeout = config.get('DEFAULT_TIMEOUT', 300)
        retries = config.get('DEFAULT_RETRIES', 2)
        
        # Use per-model timeouts if available, otherwise use default
        model_timeouts = [m.timeout for m in model_configs if m.timeout]
        timeout = max(model_timeouts) if model_timeouts else default_timeout
        
        logger.info(f"Council config - Ollama URL: {ollama_url}, Timeout: {timeout}s, Retries: {retries}")
        
        # Map churn type string to enum
        churn_type_map = {
            'edit': ChurnType.EDIT,
            'expand': ChurnType.EXPAND,
            'explore': ChurnType.EXPLORE,
        }
        churn_type = churn_type_map.get(iteration.churn_type, ChurnType.EDIT)
        logger.info(f"Churn type: {churn_type.value}")
        
        # Create churn engine instance
        logger.debug(f"Creating ChurnEngine with {len(model_names)} models")
        engine = ChurnEngine(
            models=model_names,
            ollama_url=ollama_url,
            timeout=timeout,
            retries=retries
        )
        
        # Run the churn processing (async)
        logger.info(f"Starting async churn processing for content ({len(iteration.content)} chars)")
        result = asyncio.run(
            engine.process_iteration_async(
                content=iteration.content,
                churn_type=churn_type,
                context=context
            )
        )
        logger.info(f"Churn processing completed in {result.processing_time:.1f}s")
        logger.debug(f"Got {len(result.raw_responses)} raw responses")
        
        # Save feedback
        logger.debug("Saving IterationFeedback to database")
        IterationFeedback.objects.create(
            iteration=iteration,
            synthesized_feedback=result.synthesized_feedback,
            suggested_content=result.suggested_content,
            raw_responses=result.raw_responses
        )
        
        # Update iteration with diff
        iteration.content_diff = result.content_diff
        iteration.mark_complete()
        
        logger.info(f"=== Churn iteration {iteration_id} completed successfully ===")
        return f"Churn iteration {iteration_id} completed successfully"
        
    except Exception as e:
        logger.exception(f"Churn iteration {iteration_id} failed with exception: {e}")
        iteration.mark_error(str(e))
        return f"Churn iteration {iteration_id} failed: {e}"


def run_batch_churn(
    project_id: int,
    starting_iteration_id: int,
    num_iterations: int,
    model_ids: list,
    auto_apply: bool = True,
    churn_type: str = "edit"
):
    """
    Background task to run multiple churn iterations automatically.
    
    Args:
        project_id: ID of the CreativeProject
        starting_iteration_id: ID of the ChurnIteration to start from
        num_iterations: Number of iterations to run
        model_ids: List of ModelConfig IDs to use
        auto_apply: Whether to automatically apply suggestions between iterations
        churn_type: Type of churn processing (edit/expand/explore)
    """
    try:
        project = CreativeProject.objects.get(id=project_id)
        current_iteration = ChurnIteration.objects.get(id=starting_iteration_id)
    except (CreativeProject.DoesNotExist, ChurnIteration.DoesNotExist) as e:
        return f"Error: {e}"
    
    try:
        # Get model configs
        model_configs = list(ModelConfig.objects.filter(id__in=model_ids, is_active=True))
        if len(model_configs) < 2:
            return "Not enough active models selected"
        
        model_names = [m.name for m in model_configs]
        
        # Import churn engine
        from .churn import ChurnEngine, ChurnType
        
        # Get council config from settings
        config = getattr(settings, 'COUNCIL_CONFIG', {})
        
        # Map churn type string to enum
        churn_type_map = {
            'edit': ChurnType.EDIT,
            'expand': ChurnType.EXPAND,
            'explore': ChurnType.EXPLORE,
        }
        churn_enum = churn_type_map.get(churn_type, ChurnType.EDIT)
        
        # Use per-model timeouts if available, otherwise use default
        default_timeout = config.get('DEFAULT_TIMEOUT', 300)
        model_timeouts = [m.timeout for m in model_configs if m.timeout]
        timeout = max(model_timeouts) if model_timeouts else default_timeout
        
        # Create churn engine instance
        engine = ChurnEngine(
            models=model_names,
            ollama_url=config.get('OLLAMA_URL', 'http://localhost:11434'),
            timeout=timeout,
            retries=config.get('DEFAULT_RETRIES', 2)
        )
        
        # Run all iterations in a single async context
        result = asyncio.run(
            run_batch_churn_async(
                engine, project, current_iteration, num_iterations,
                model_configs, auto_apply, churn_type, churn_enum
            )
        )
        return result
        
    except Exception as e:
        return f"Batch churn failed: {e}"


async def run_batch_churn_async(
    engine, project, current_iteration, num_iterations,
    model_configs, churn_type, churn_enum, auto_apply=True
):
    """
    Async helper for running batch churn iterations.
    """
    completed_iterations = 0
    
    for i in range(num_iterations):
        # Determine content for this iteration (access ORM via sync_to_async)
        @sync_to_async
        def get_iteration_content(iteration, auto_apply_flag):
            try:
                if auto_apply_flag and hasattr(iteration, 'feedback') and iteration.feedback.suggested_content:
                    return iteration.feedback.suggested_content
            except Exception:
                pass
            return iteration.content
        
        content = await get_iteration_content(current_iteration, auto_apply)
        
        # Create new iteration
        new_iteration = await sync_to_async(ChurnIteration.objects.create)(
            project=project,
            parent=current_iteration,
            content=content,
            churn_type=churn_type,
            iteration_number=current_iteration.iteration_number + 1,
            status=ChurnIteration.Status.PROCESSING
        )
        await sync_to_async(new_iteration.models_used.set)(model_configs)
        
        try:
            # Run the churn processing
            result = await engine.process_iteration_async(
                content=content,
                churn_type=churn_enum,
                context=""
            )
            
            # Save feedback
            await sync_to_async(IterationFeedback.objects.create)(
                iteration=new_iteration,
                synthesized_feedback=result.synthesized_feedback,
                suggested_content=result.suggested_content,
                raw_responses=result.raw_responses
            )
            
            # Update iteration with diff
            new_iteration.content_diff = result.content_diff
            await sync_to_async(new_iteration.mark_complete)()
            
            completed_iterations += 1
            current_iteration = new_iteration
            
        except Exception as e:
            await sync_to_async(new_iteration.mark_error)(str(e))
            return f"Batch churn failed at iteration {i+1}: {e}"
    
    return f"Batch churn completed: {completed_iterations}/{num_iterations} iterations"
