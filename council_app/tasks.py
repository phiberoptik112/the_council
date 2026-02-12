"""
Background tasks for running council queries.
Uses Django-Q2 for async task execution.
"""
import asyncio
import logging
from django.conf import settings
from django.utils import timezone
from asgiref.sync import sync_to_async

from .models import (
    Query, Response, Vote, ModelConfig, ChurnIteration, IterationFeedback,
    ChurnConfig, CreativeProject, QueryEvent, ReportSection, ReportOutline,
    PDFDocument, PDFPage,
)

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
    
    # --- Response Validation ---
    # Validate responses before saving/voting to filter refusals and off-topic answers
    valid_responses, flagged_responses, validation_details = council.validate_responses(
        responses, query.prompt
    )
    
    # Log validation results for each flagged response
    for flagged_resp in flagged_responses:
        detail = validation_details.get(flagged_resp.model, {})
        reasons = '; '.join(detail.get('reasons', ['Unknown']))
        await log_query_event(
            query.id,
            QueryEvent.EventType.MODEL_ERROR,
            model_name=flagged_resp.model,
            message=f"Response flagged by validation: {reasons}",
            raw_data={
                'validation': detail,
                'response_preview': flagged_resp.response[:200],
            }
        )
    
    # Log validation summary
    using_fallback = len(valid_responses) == len(responses) and len(flagged_responses) > 0
    await log_query_event(
        query.id,
        QueryEvent.EventType.QUERY_STARTED,
        message=(
            f"Validation complete: {len(valid_responses)} valid, "
            f"{len(flagged_responses)} flagged out of {len(responses)} total"
            + (" (fallback: using all responses)" if using_fallback else "")
        ),
        raw_data={
            'valid_count': len(valid_responses),
            'flagged_count': len(flagged_responses),
            'total_count': len(responses),
            'using_fallback': using_fallback,
        }
    )
    
    # Save ALL responses to database (both valid and flagged) with validation status
    labels = council.LABELS
    db_responses = []
    
    for i, resp in enumerate(responses):
        if resp.error:
            continue
            
        model_config = model_map.get(resp.model)
        if not model_config:
            continue
        
        detail = validation_details.get(resp.model, {})
        is_valid = detail.get('is_valid', True)
        reasons = '; '.join(detail.get('reasons', []))
        
        db_response = await sync_to_async(Response.objects.create)(
            query=query,
            model=model_config,
            content=resp.response,
            label=labels[i],
            response_time=resp.response_time,
            is_validated=is_valid,
            validation_notes=reasons,
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
        message=f"Starting voting phase with {len(valid_responses)} validated responses"
    )
    
    # Conduct voting on validated responses only (or all if fallback)
    vote_results = await council.conduct_vote_async(query.prompt, valid_responses, callback=event_callback)
    
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
    
    # Build mapping from voting labels back to model names.
    # Voting was done on valid_responses, so voting label 'A' = valid_responses[0], etc.
    voting_label_to_model = {}
    for i, resp in enumerate(valid_responses):
        if i < len(labels):
            voting_label_to_model[labels[i]] = resp.model
    
    # Map voting scores to model names for correct DB matching
    raw_scores = vote_results.get('scores', {})
    model_scores = {}
    for voting_label, score in raw_scores.items():
        model_name = voting_label_to_model.get(voting_label)
        if model_name:
            model_scores[model_name] = score
    
    # Determine the winning model from voting results
    winner_model_response = vote_results.get('winner')
    winner_model_name = winner_model_response.model if winner_model_response else None
    
    # Update response scores and determine winner
    for db_response in db_responses:
        db_response.score = model_scores.get(db_response.model.name, 0)
        db_response.is_winner = (db_response.model.name == winner_model_name)
        await sync_to_async(db_response.save)()
        
        # Update model statistics
        await sync_to_async(db_response.model.update_stats)(
            won=db_response.is_winner,
            response_time=db_response.response_time
        )
    
    # Find and set winner
    winner_response = next((r for r in db_responses if r.is_winner), None)
    
    # Determine the winning label for logging
    winner_label = winner_response.label if winner_response else 'N/A'
    
    # Handle synthesis/debate modes (use valid_responses for these too)
    if mode == CouncilMode.SYNTHESIS and winner_response:
        synthesis_result = await council.synthesize_response_async(
            query.prompt, valid_responses
        )
        if synthesis_result and not synthesis_result.error:
            winner_response.content = synthesis_result.response
            await sync_to_async(winner_response.save)()
    
    elif mode == CouncilMode.DEBATE and winner_response:
        winning_model_response = next(
            (r for r in valid_responses if r.model == winner_response.model.name), 
            None
        )
        if winning_model_response:
            debate_result = await council.debate_and_refine_async(
                query.prompt, winning_model_response, valid_responses
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
        raw_data={
            'scores': {k: v for k, v in model_scores.items()},
            'winner_label': winner_label,
            'validated_count': len(valid_responses),
            'flagged_count': len(flagged_responses),
        }
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
        
        # Save initial debug_info so polling UI can show models being queried
        iteration.debug_info = {
            "models_queried": model_names,
            "responses": [],
            "validation_summary": {
                "total_responses": 0,
                "valid_count": 0,
                "flagged_count": 0,
            },
        }
        iteration.save(update_fields=["debug_info"])
        
        # Import churn engine
        from .churn import ChurnEngine, ChurnType, ChurnCancelled
        
        # Get config early for callback and engine
        config = ChurnConfig.get_effective_config()
        debug_full = config.get('CHURN_DEBUG_FULL_RESPONSES', False)
        debug_max_chars = config.get('CHURN_DEBUG_RESPONSE_MAX_CHARS', 500)
        
        def save_response_callback(model, stage, data):
            """Persist each model response as it completes for live display."""
            if stage == "success" and data:
                resp_text = data.response if data.response else None
                if resp_text and not debug_full:
                    resp_text = resp_text[:debug_max_chars]
                entry = {
                    "model": data.model,
                    "response": resp_text,
                    "response_length": len(data.response) if data.response else 0,
                    "response_time": data.response_time,
                    "error": data.error,
                    "is_empty": not (data.response and data.response.strip()),
                    "validation": {},
                }
            elif stage in ("error", "timeout"):
                entry = {
                    "model": model,
                    "response": None,
                    "response_length": 0,
                    "response_time": 0,
                    "error": str(data) if data else f"{stage}",
                    "is_empty": True,
                    "validation": {},
                }
            else:
                return
            try:
                iteration.refresh_from_db()
                debug_info = dict(iteration.debug_info or {})
                responses = list(debug_info.get("responses", []))
                responses.append(entry)
                debug_info["responses"] = responses
                debug_info["validation_summary"] = debug_info.get("validation_summary", {})
                debug_info["validation_summary"]["total_responses"] = len(responses)
                iteration.debug_info = debug_info
                iteration.save(update_fields=["debug_info"])
            except Exception as e:
                logger.warning(f"Failed to persist live response from {model}: {e}")
        
        def cancel_check():
            """Check if user requested cancellation."""
            iteration.refresh_from_db()
            return iteration.cancel_requested
        
        # Config already loaded above
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
            'report': ChurnType.REPORT,
        }
        churn_type = churn_type_map.get(iteration.churn_type, ChurnType.EDIT)
        logger.info(f"Churn type: {churn_type.value}")
        
        # Create churn engine instance with performance options
        logger.debug(f"Creating ChurnEngine with {len(model_names)} models")
        engine = ChurnEngine(
            models=model_names,
            ollama_url=ollama_url,
            timeout=timeout,
            retries=retries,
            num_predict=config.get('NUM_PREDICT'),
            keep_alive=config.get('KEEP_ALIVE'),
            num_ctx=config.get('NUM_CTX'),
            sequential=config.get('CHURN_SEQUENTIAL_MODELS', False),
            max_content_chars=config.get('CHURN_MAX_CONTENT_CHARS'),
            max_synthesis_response_chars=config.get('CHURN_MAX_SYNTHESIS_RESPONSE_CHARS'),
            use_streaming=config.get('CHURN_USE_STREAMING', False),
            debug_full_responses=debug_full,
            debug_response_max_chars=debug_max_chars,
        )
        
        # Run the churn processing (async)
        logger.info(f"Starting async churn processing for content ({len(iteration.content)} chars)")
        try:
            result = asyncio.run(
                engine.process_iteration_async(
                    content=iteration.content,
                    churn_type=churn_type,
                    context=context,
                    callback=save_response_callback,
                    cancel_check=cancel_check,
                )
            )
            logger.info(f"Churn processing completed in {result.processing_time:.1f}s")
            logger.debug(f"Got {len(result.raw_responses)} raw responses")
            
            # Store debug info if available (even on success, for troubleshooting)
            if result.debug_info:
                iteration.debug_info = result.debug_info
                iteration.save()
                logger.debug("Stored debug info in iteration")
            
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
            
        except ChurnCancelled as e:
            logger.info(f"Churn iteration {iteration_id} cancelled by user")
            # Keep existing debug_info (partial responses already saved by callback)
            iteration.mark_error("Cancelled by user")
            return f"Churn iteration {iteration_id} cancelled"
        
        except ValueError as e:
            # Check if exception has debug_info attached
            debug_info = getattr(e, 'debug_info', None)
            if debug_info:
                iteration.debug_info = debug_info
                iteration.save()
                logger.debug("Stored debug info from validation failure")
            
            error_message = str(e)
            logger.exception(f"Churn iteration {iteration_id} failed with validation error: {error_message}")
            iteration.mark_error(error_message)
            return f"Churn iteration {iteration_id} failed: {error_message}"
        
    except Exception as e:
        # For other exceptions, try to build basic debug info
        debug_info = {
            "models_queried": model_names if 'model_names' in locals() else [],
            "error_type": type(e).__name__,
            "error_message": str(e),
            "error_details": f"Unexpected error during churn processing: {e}"
        }
        iteration.debug_info = debug_info
        iteration.save()
        
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
        
        # Get config (ChurnConfig DB overrides COUNCIL_CONFIG)
        config = ChurnConfig.get_effective_config()
        
        # Map churn type string to enum
        churn_type_map = {
            'edit': ChurnType.EDIT,
            'expand': ChurnType.EXPAND,
            'explore': ChurnType.EXPLORE,
            'report': ChurnType.REPORT,
        }
        churn_enum = churn_type_map.get(churn_type, ChurnType.EDIT)
        
        # Use per-model timeouts if available, otherwise use default
        default_timeout = config.get('DEFAULT_TIMEOUT', 300)
        model_timeouts = [m.timeout for m in model_configs if m.timeout]
        timeout = max(model_timeouts) if model_timeouts else default_timeout
        
        # Create churn engine instance with performance options
        engine = ChurnEngine(
            models=model_names,
            ollama_url=config.get('OLLAMA_URL', 'http://localhost:11434'),
            timeout=timeout,
            retries=config.get('DEFAULT_RETRIES', 2),
            num_predict=config.get('NUM_PREDICT'),
            keep_alive=config.get('KEEP_ALIVE'),
            num_ctx=config.get('NUM_CTX'),
            sequential=config.get('CHURN_SEQUENTIAL_MODELS', False),
            max_content_chars=config.get('CHURN_MAX_CONTENT_CHARS'),
            max_synthesis_response_chars=config.get('CHURN_MAX_SYNTHESIS_RESPONSE_CHARS'),
            use_streaming=config.get('CHURN_USE_STREAMING', False),
        )
        
        # Run all iterations in a single async context
        result = asyncio.run(
            run_batch_churn_async(
                engine, project, current_iteration, num_iterations,
                model_configs, churn_type, churn_enum, auto_apply
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


# =============================================================================
# TECHNICAL REPORT REVIEWER TASKS
# =============================================================================

def run_report_section_churn(section_id: int, model_ids: list):
    """
    Background task to process a single report section through the council.

    Args:
        section_id: ID of the ReportSection object
        model_ids: List of ModelConfig IDs to use
    """
    logger.info(f"=== Starting report section churn for section {section_id} ===")
    logger.debug(f"Model IDs: {model_ids}")

    try:
        section = ReportSection.objects.select_related(
            'report_outline',
            'report_outline__knowledgebase',
            'report_outline__project'
        ).get(id=section_id)
        logger.debug(f"Found section: '{section.section_title}' in project: {section.report_outline.project.title}")
    except ReportSection.DoesNotExist:
        logger.error(f"ReportSection {section_id} not found in database")
        return f"ReportSection {section_id} not found"

    try:
        # Get model configs
        model_configs = list(ModelConfig.objects.filter(id__in=model_ids, is_active=True))
        logger.info(f"Found {len(model_configs)} active models from {len(model_ids)} requested")

        if len(model_configs) < 2:
            logger.error(f"Not enough active models: {len(model_configs)} < 2")
            section.mark_error("Not enough active models selected")
            return "Not enough models"

        model_names = [m.name for m in model_configs]
        logger.info(f"Using models: {model_names}")

        # Update status to in progress
        section.mark_in_progress()
        logger.debug("Section status set to IN_PROGRESS")

        # Initial debug_info for live display
        section.debug_info = {
            "models_queried": model_names,
            "responses": [],
            "validation_summary": {"total_responses": 0, "valid_count": 0, "flagged_count": 0},
        }
        section.save(update_fields=["debug_info"])

        # Import report churn engine
        from .report_churn import ReportChurnEngine
        from .churn import ChurnCancelled

        config = ChurnConfig.get_effective_config()
        debug_full = config.get('CHURN_DEBUG_FULL_RESPONSES', False)
        debug_max_chars = config.get('CHURN_DEBUG_RESPONSE_MAX_CHARS', 500)

        def save_response_callback(model, stage, data):
            """Persist each model response for live display."""
            if stage == "success" and data:
                resp_text = data.response if data.response else None
                if resp_text and not debug_full:
                    resp_text = resp_text[:debug_max_chars]
                entry = {
                    "model": data.model,
                    "response": resp_text,
                    "response_length": len(data.response) if data.response else 0,
                    "response_time": data.response_time,
                    "error": data.error,
                    "is_empty": not (data.response and data.response.strip()),
                    "validation": {},
                }
            elif stage in ("error", "timeout"):
                entry = {
                    "model": model,
                    "response": None,
                    "response_length": 0,
                    "response_time": 0,
                    "error": str(data) if data else str(stage),
                    "is_empty": True,
                    "validation": {},
                }
            else:
                return
            try:
                section.refresh_from_db()
                debug_info = dict(section.debug_info or {})
                responses = list(debug_info.get("responses", []))
                responses.append(entry)
                debug_info["responses"] = responses
                debug_info["validation_summary"] = debug_info.get("validation_summary", {})
                debug_info["validation_summary"]["total_responses"] = len(responses)
                section.debug_info = debug_info
                section.save(update_fields=["debug_info"])
            except Exception as e:
                logger.warning(f"Failed to persist live response from {model}: {e}")

        def cancel_check():
            section.refresh_from_db()
            return section.cancel_requested

        # Council config from ChurnConfig
        ollama_url = config.get('OLLAMA_URL', 'http://localhost:11434')
        default_timeout = config.get('DEFAULT_TIMEOUT', 300)
        retries = config.get('DEFAULT_RETRIES', 2)
        model_timeouts = [m.timeout for m in model_configs if m.timeout]
        timeout = max(model_timeouts) if model_timeouts else default_timeout
        logger.info(f"Council config - Ollama URL: {ollama_url}, Timeout: {timeout}s, Retries: {retries}")

        # Get knowledgebase content
        kb_content = ""
        outline = section.report_outline
        if outline.knowledgebase:
            kb_content = outline.knowledgebase.get_relevant_context(
                section.section_title,
                section.original_content or section.current_content
            )
            logger.debug(f"Knowledgebase content: {len(kb_content)} chars")

        # Get preceding approved sections for context
        preceding = ReportSection.objects.filter(
            report_outline=outline,
            status=ReportSection.Status.APPROVED,
            order__lt=section.order
        ).order_by('order')[:3]

        preceding_sections = [
            {'title': s.section_title, 'content': s.current_content}
            for s in preceding
        ]
        logger.debug(f"Preceding approved sections: {len(preceding_sections)}")

        # Create engine and process
        logger.debug(f"Creating ReportChurnEngine with {len(model_names)} models")
        engine = ReportChurnEngine(
            models=model_names,
            ollama_url=ollama_url,
            timeout=timeout,
            retries=retries,
            knowledgebase_content=kb_content
        )

        # Run the section review (async)
        content_to_review = section.current_content or section.original_content
        logger.info(f"Starting async section review for '{section.section_title}' ({len(content_to_review)} chars)")

        try:
            result = asyncio.run(
                engine.process_section_async(
                    section_id=section.section_id,
                    section_title=section.section_title,
                    section_content=content_to_review,
                    knowledge_context=kb_content,
                    full_outline=outline.raw_outline,
                    preceding_sections=preceding_sections,
                    callback=save_response_callback,
                    cancel_check=cancel_check,
                )
            )
        except ChurnCancelled:
            logger.info(f"Report section {section_id} cancelled by user")
            section.mark_error("Cancelled by user")
            return f"Section '{section.section_title}' cancelled"
        logger.info(f"Section review completed in {result.processing_time:.1f}s")
        logger.debug(f"Compliance score: {result.compliance_score}")

        # Update section with results
        section.current_content = result.revised_content
        section.content_diff = result.content_diff
        section.council_feedback = {
            'synthesized_feedback': result.synthesized_feedback,
            'processing_time': result.processing_time,
        }
        section.compliance_score = result.compliance_score
        section.raw_responses = result.raw_responses
        section.knowledge_context = kb_content
        section.iteration_count += 1
        section.status = ReportSection.Status.REVIEW
        section.error_message = ''
        section.save()

        logger.info(f"=== Report section '{section.section_title}' processed successfully ===")
        return f"Section '{section.section_title}' processed successfully"

    except Exception as e:
        logger.exception(f"Report section {section_id} failed with exception: {e}")
        section.mark_error(str(e))
        return f"Report section {section_id} failed: {e}"


def run_full_report_churn(report_outline_id: int, model_ids: list, auto_advance: bool = False):
    """
    Background task to process all sections of a report outline.

    Args:
        report_outline_id: ID of the ReportOutline object
        model_ids: List of ModelConfig IDs to use
        auto_advance: If True, process all sections sequentially.
                      If False, stop after first pending section.
    """
    logger.info(f"=== Starting full report churn for outline {report_outline_id} ===")
    logger.debug(f"Model IDs: {model_ids}, Auto-advance: {auto_advance}")

    try:
        outline = ReportOutline.objects.select_related(
            'project', 'knowledgebase'
        ).get(id=report_outline_id)
        logger.debug(f"Found outline for project: {outline.project.title}")
    except ReportOutline.DoesNotExist:
        logger.error(f"ReportOutline {report_outline_id} not found")
        return f"ReportOutline {report_outline_id} not found"

    try:
        # Ensure sections exist from parsed_sections
        for section_data in outline.parsed_sections:
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

        # Get sections to process
        sections = ReportSection.objects.filter(
            report_outline=outline,
            status__in=[
                ReportSection.Status.PENDING,
                ReportSection.Status.NEEDS_REVISION,
            ]
        ).order_by('order')

        if not sections.exists():
            logger.info("No pending sections to process")
            return "No pending sections to process"

        processed_count = 0
        for section in sections:
            logger.info(f"Processing section: '{section.section_title}'")
            result = run_report_section_churn(section.id, model_ids)
            logger.info(f"Section result: {result}")
            processed_count += 1

            if not auto_advance:
                logger.info("Auto-advance disabled, stopping after first section")
                break

        return f"Report churn completed: {processed_count} sections processed"

    except Exception as e:
        logger.exception(f"Full report churn failed: {e}")
        return f"Full report churn failed: {e}"


# =============================================================================
# PDF TO MARKDOWN PROCESSING TASKS
# =============================================================================

def process_pdf_document(document_id: int):
    """
    Background task to process a PDF document page-by-page through a vision model.

    This is a SINGLE-MODEL task (not a council process). Each page is:
    1. Rendered to a PNG image at controlled DPI
    2. Sent to an Ollama vision model for markdown extraction
    3. Stored in the database (PDFPage)
    4. Stored in ChromaDB for vector search

    Pages are processed sequentially to conserve RAM on Raspberry Pi.

    Args:
        document_id: ID of the PDFDocument to process
    """
    logger.info(f"=== Starting PDF processing for document {document_id} ===")

    try:
        document = PDFDocument.objects.get(id=document_id)
    except PDFDocument.DoesNotExist:
        logger.error(f"PDFDocument {document_id} not found")
        return f"PDFDocument {document_id} not found"

    try:
        from .pdf_processor import PDFPageProcessor

        # Mark as processing
        document.mark_processing()

        # Resolve the file path
        pdf_path = document.file.path
        logger.info(f"Processing PDF: {pdf_path}")

        # Get page count
        total_pages = PDFPageProcessor.extract_page_count(pdf_path)
        document.total_pages = total_pages
        document.save(update_fields=['total_pages'])
        logger.info(f"PDF has {total_pages} pages")

        # Create PDFPage records for all pages
        for page_num in range(total_pages):
            PDFPage.objects.get_or_create(
                document=document,
                page_number=page_num,
                defaults={'status': PDFPage.Status.PENDING},
            )

        # Determine vision model name
        model_name = None
        if document.model:
            model_name = document.model.name
        if not model_name:
            pdf_config = getattr(settings, 'PDF_PROCESSING', {})
            model_name = pdf_config.get('DEFAULT_VISION_MODEL', 'moondream')

        # Build collection name for ChromaDB
        import re
        slug = re.sub(r'[^a-zA-Z0-9]', '_', document.title.lower())[:30]
        collection_name = f"pdf_{document.id}_{slug}"
        document.chromadb_collection = collection_name
        document.save(update_fields=['chromadb_collection'])

        # Create processor
        processor = PDFPageProcessor(model_name=model_name)

        # Metadata for ChromaDB storage
        base_metadata = {
            'document_id': document.id,
            'title': document.title,
            'filename': document.file.name,
        }

        # Process pages sequentially (RAM-friendly)
        async def process_all_pages():
            for page_num in range(total_pages):
                page_obj = PDFPage.objects.get(
                    document=document, page_number=page_num
                )

                # Skip already-completed pages (for retry scenarios)
                if page_obj.status == PDFPage.Status.COMPLETE:
                    logger.info(f"Page {page_num} already complete, skipping")
                    continue

                page_obj.mark_processing()

                try:
                    import time as _time
                    start = _time.time()

                    markdown = await processor.process_single_page(
                        pdf_path=pdf_path,
                        page_num=page_num,
                        collection_name=collection_name,
                        metadata=base_metadata,
                    )

                    elapsed = _time.time() - start
                    page_obj.mark_complete(
                        markdown=markdown,
                        processing_time=elapsed,
                    )

                    # Update document progress
                    document.processed_pages = page_num + 1
                    document.save(update_fields=['processed_pages'])

                    logger.info(
                        f"Page {page_num + 1}/{total_pages} complete "
                        f"({elapsed:.1f}s, {len(markdown)} chars)"
                    )

                except Exception as page_error:
                    logger.exception(f"Error processing page {page_num}: {page_error}")
                    page_obj.mark_error(str(page_error))
                    # Continue to next page instead of stopping entirely
                    document.processed_pages = page_num + 1
                    document.save(update_fields=['processed_pages'])

        # Run the async processing
        asyncio.run(process_all_pages())

        # Check if all pages completed successfully
        error_pages = document.pages.filter(status=PDFPage.Status.ERROR).count()
        complete_pages = document.pages.filter(status=PDFPage.Status.COMPLETE).count()

        if error_pages > 0:
            document.mark_error(
                f"{error_pages} of {total_pages} pages had errors. "
                f"{complete_pages} pages processed successfully."
            )
            # Still mark as complete if most pages succeeded
            if complete_pages > 0:
                document.status = PDFDocument.Status.COMPLETE
                document.error_message = (
                    f"{error_pages} of {total_pages} pages had errors. "
                    f"{complete_pages} pages processed successfully."
                )
                document.save(update_fields=['status', 'error_message'])
        else:
            document.mark_complete()

        logger.info(
            f"=== PDF processing complete: {complete_pages}/{total_pages} pages, "
            f"{error_pages} errors ==="
        )
        return (
            f"PDF '{document.title}' processed: "
            f"{complete_pages}/{total_pages} pages complete"
        )

    except Exception as e:
        logger.exception(f"PDF processing failed for document {document_id}: {e}")
        document.mark_error(str(e))
        return f"PDF processing failed: {e}"


# =============================================================================
# STUCK TASK CLEANUP
# =============================================================================

def cleanup_stuck_tasks(stale_minutes: int = 45):
    """
    Periodic cleanup task that detects and marks stale processing tasks as errors.

    When Django-Q2 kills a task for exceeding its timeout (via SIGKILL), the
    exception handler in the task function never runs, leaving the database
    record stuck in a "processing" / "gathering" / "in_progress" state forever.
    This task finds those orphaned records and marks them as errors so the user
    can see what happened and retry.

    Args:
        stale_minutes: Number of minutes after which a processing task is
                       considered stuck (default 45).
    """
    from datetime import timedelta

    cutoff = timezone.now() - timedelta(minutes=stale_minutes)
    cleaned = 0

    # --- ChurnIteration stuck in 'processing' ---
    stuck_iterations = ChurnIteration.objects.filter(
        status=ChurnIteration.Status.PROCESSING,
        created_at__lt=cutoff,
    )
    for iteration in stuck_iterations:
        logger.warning(
            f"Cleaning up stuck ChurnIteration {iteration.id} "
            f"(created {iteration.created_at}, project: {iteration.project.title})"
        )
        iteration.mark_error(
            "Task timed out or was terminated unexpectedly. "
            "Please retry the iteration."
        )
        cleaned += 1

    # --- Query stuck in 'gathering' or 'voting' ---
    stuck_queries = Query.objects.filter(
        status__in=[Query.Status.GATHERING, Query.Status.VOTING],
        created_at__lt=cutoff,
    )
    for query in stuck_queries:
        logger.warning(
            f"Cleaning up stuck Query {query.id} "
            f"(status: {query.status}, created {query.created_at})"
        )
        query.mark_error(
            "Task timed out or was terminated unexpectedly. "
            "Please submit your query again."
        )
        cleaned += 1

    # --- ReportSection stuck in 'in_progress' ---
    stuck_sections = ReportSection.objects.filter(
        status=ReportSection.Status.IN_PROGRESS,
        updated_at__lt=cutoff,
    )
    for section in stuck_sections:
        logger.warning(
            f"Cleaning up stuck ReportSection {section.id} "
            f"(title: '{section.section_title}', updated {section.updated_at})"
        )
        section.mark_error(
            "Task timed out or was terminated unexpectedly. "
            "Please retry this section."
        )
        cleaned += 1

    # --- PDFDocument stuck in 'processing' ---
    stuck_pdfs = PDFDocument.objects.filter(
        status=PDFDocument.Status.PROCESSING,
        updated_at__lt=cutoff,
    )
    for pdf_doc in stuck_pdfs:
        logger.warning(
            f"Cleaning up stuck PDFDocument {pdf_doc.id} "
            f"(title: '{pdf_doc.title}', updated {pdf_doc.updated_at})"
        )
        pdf_doc.mark_error(
            "Task timed out or was terminated unexpectedly. "
            "Please retry processing."
        )
        cleaned += 1

    if cleaned:
        logger.info(f"Stuck-task cleanup: marked {cleaned} stuck records as errors")
    else:
        logger.debug("Stuck-task cleanup: no stuck records found")

    return f"Cleaned up {cleaned} stuck tasks"
