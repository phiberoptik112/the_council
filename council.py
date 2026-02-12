#!/usr/bin/env python3
"""
Multi-Model Council System
Orchestrates multiple LLMs to vote on best responses

Features:
- Async parallel model queries
- Self-voting bias prevention
- Multiple council modes (ranked, majority, synthesis, debate)
- Robust vote parsing with JSON fallback
- Timeout and retry handling
"""

import asyncio
import aiohttp
import json
import logging
import re
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import time

# Set up logger for debugging
logger = logging.getLogger('council')


class CouncilMode(Enum):
    RANKED = "ranked"           # Borda count ranking (default)
    MAJORITY = "majority"       # Simple majority vote
    SYNTHESIS = "synthesis"     # Combine best elements into new response
    DEBATE = "debate"           # Models critique and refine winner


@dataclass
class ModelResponse:
    """Represents a response from a single model"""
    model: str
    response: str
    response_time: float = 0.0
    error: Optional[str] = None


@dataclass
class VoteResult:
    """Represents voting results"""
    scores: Dict[str, int]
    winner: ModelResponse
    winner_label: str
    all_responses: List[ModelResponse]
    all_votes: List[Dict[str, Any]]
    mode: CouncilMode


class ResponseValidator:
    """
    Validates that model responses meet minimum quality thresholds.
    
    Checks:
    1. Refusal detection - catches common LLM refusal/decline phrases
    2. Minimum length - ensures responses are substantive
    3. Prompt relevance - checks keyword overlap with the original prompt
    """
    
    # Common refusal patterns (compiled for performance)
    REFUSAL_PATTERNS = [
        re.compile(r"i[''']?m sorry.{0,40}(?:can[''']?t|cannot|unable to|won[''']?t)", re.IGNORECASE),
        re.compile(r"i (?:can[''']?t|cannot|won[''']?t|am unable to) (?:assist|help|provide|respond|answer|fulfill|comply)", re.IGNORECASE),
        re.compile(r"as an ai.{0,30}(?:can[''']?t|cannot|unable|not able|don[''']?t)", re.IGNORECASE),
        re.compile(r"i[''']?m not able to (?:assist|help|provide|respond|answer)", re.IGNORECASE),
        re.compile(r"(?:this|that|your) (?:request|query|question|prompt) (?:is|goes|violates|falls)", re.IGNORECASE),
        re.compile(r"against my (?:guidelines|programming|policy|rules|principles)", re.IGNORECASE),
        re.compile(r"i (?:must|have to) (?:decline|refuse|refrain)", re.IGNORECASE),
        re.compile(r"i[''']?m (?:not )?(?:designed|programmed|allowed) to", re.IGNORECASE),
        re.compile(r"(?:can[''']?t|cannot|unable to) (?:generate|create|produce|write) (?:that|this|such)", re.IGNORECASE),
        re.compile(r"if you have (?:any )?other questions.{0,20}feel free", re.IGNORECASE),
    ]
    
    # Minimum character count for a substantive response
    MIN_RESPONSE_LENGTH = 50
    
    # Stop words excluded from relevance checking
    STOP_WORDS = frozenset({
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
        'her', 'was', 'one', 'our', 'out', 'has', 'have', 'with', 'this', 'that',
        'from', 'what', 'how', 'why', 'when', 'where', 'which', 'who', 'will',
        'would', 'could', 'should', 'their', 'there', 'been', 'being', 'does',
        'about', 'into', 'more', 'other', 'than', 'then', 'them', 'these', 'some',
        'each', 'any', 'also', 'just', 'like', 'make', 'know', 'take', 'come',
        'give', 'most', 'find', 'here', 'thing', 'many', 'well', 'only', 'tell',
        'very', 'even', 'back', 'after', 'use', 'two', 'way', 'look', 'help',
        'first', 'new', 'because', 'day', 'get', 'got', 'say', 'she', 'him',
        'his', 'its', 'may', 'still', 'over', 'such', 'now', 'own', 'keep',
        'let', 'begin', 'seem', 'talk', 'need', 'put', 'try', 'ask', 'too',
        'explain', 'describe', 'please', 'write', 'what', 'list',
    })
    
    @classmethod
    def validate_response(cls, response_text: str, original_prompt: str) -> dict:
        """
        Validate a single response against minimum quality thresholds.
        
        Args:
            response_text: The model's response content
            original_prompt: The original user prompt
            
        Returns:
            dict with:
                is_valid: bool - whether response passes all checks
                reasons: list[str] - failure reasons (empty if valid)
                is_refusal: bool - whether a refusal pattern was detected
                relevance_score: float - 0.0 to 1.0 keyword overlap score
        """
        reasons = []
        
        # Check 1: Refusal detection
        is_refusal = cls._detect_refusal(response_text)
        if is_refusal:
            reasons.append("Response appears to be a refusal or decline to answer")
        
        # Check 2: Minimum length
        stripped = response_text.strip()
        if len(stripped) < cls.MIN_RESPONSE_LENGTH:
            reasons.append(
                f"Response too short ({len(stripped)} chars, minimum {cls.MIN_RESPONSE_LENGTH})"
            )
        
        # Check 3: Prompt relevance
        relevance_score = cls._check_relevance(response_text, original_prompt)
        if relevance_score < 0.1:
            reasons.append(
                f"Response has very low relevance to the prompt (score: {relevance_score:.2f})"
            )
        
        return {
            'is_valid': len(reasons) == 0,
            'reasons': reasons,
            'is_refusal': is_refusal,
            'relevance_score': relevance_score,
        }
    
    @classmethod
    def _detect_refusal(cls, text: str) -> bool:
        """Check if the response matches known refusal patterns."""
        for pattern in cls.REFUSAL_PATTERNS:
            if pattern.search(text):
                return True
        return False
    
    @classmethod
    def _check_relevance(cls, response: str, prompt: str) -> float:
        """
        Calculate keyword overlap between prompt and response.
        
        Returns a float from 0.0 (no overlap) to 1.0 (all prompt keywords found).
        """
        # Extract significant words from the prompt (3+ chars, not stop words)
        prompt_words = set(
            word.lower() for word in re.findall(r'\b\w+\b', prompt)
            if len(word) >= 3 and word.lower() not in cls.STOP_WORDS
        )
        
        if not prompt_words:
            # Can't assess relevance without meaningful prompt keywords
            return 1.0
        
        response_lower = response.lower()
        matched = sum(1 for word in prompt_words if word in response_lower)
        return matched / len(prompt_words)


class LLMCouncil:
    """
    Multi-model council that queries multiple LLMs and votes on best response.
    
    Features:
    - Parallel async queries for speed
    - Self-vote exclusion to prevent bias
    - Response validation to filter refusals and off-topic answers
    - Multiple voting modes
    - Robust error handling with retries
    """
    
    LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    def __init__(
        self,
        models: List[str],
        ollama_url: str = "http://localhost:11434",
        timeout: int = 600,  # 10 minutes - extended timeout for slow models on limited RAM
        retries: int = 2,
        exclude_self_vote: bool = True,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        stream: bool = False
    ):
        self.models = models
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.retries = retries
        self.exclude_self_vote = exclude_self_vote
        self.options = options or {}
        self.keep_alive = keep_alive
        self.stream = stream
        
    async def check_model_available(self, session: aiohttp.ClientSession, model: str) -> bool:
        """Check if a model is available in Ollama"""
        logger.debug(f"Checking if model '{model}' is available at {self.ollama_url}")
        try:
            async with session.get(f"{self.ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    available_models = [m.get('name', '') for m in data.get('models', [])]
                    logger.debug(f"Available models in Ollama: {available_models}")
                    # Check both exact match and base name match
                    is_available = any(model in m or m.startswith(model.split(':')[0]) for m in available_models)
                    logger.info(f"Model '{model}' availability: {is_available}")
                    return is_available
                else:
                    logger.warning(f"Ollama /api/tags returned status {resp.status}")
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Cannot connect to Ollama at {self.ollama_url}: {e}")
        except Exception as e:
            logger.exception(f"Error checking model availability: {e}")
        return False
    
    async def query_model_async(
        self,
        session: aiohttp.ClientSession,
        model: str,
        prompt: str,
        callback: Optional[callable] = None
    ) -> ModelResponse:
        """Query a single model with retry logic"""
        start_time = time.time()
        last_error = None
        api_url = f"{self.ollama_url}/api/generate"
        
        logger.debug(f"Querying model '{model}' at {api_url}")
        logger.debug(f"Prompt length: {len(prompt)} chars, Timeout: {self.timeout}s")
        
        # Notify that we're starting to query this model
        if callback:
            callback(model, "querying", None)
        
        for attempt in range(self.retries + 1):
            if attempt > 0:
                logger.debug(f"Retry attempt {attempt} for model '{model}'")
                if callback:
                    callback(model, "retry", f"Attempt {attempt + 1}")
            
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                request_payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": self.stream
                }
                if self.options:
                    request_payload["options"] = {
                        k: v for k, v in self.options.items()
                        if v is not None
                    }
                if self.keep_alive:
                    request_payload["keep_alive"] = self.keep_alive
                logger.debug(f"Sending request to '{model}' at {api_url}: {json.dumps(request_payload, indent=2)[:500]}...")
                
                async with session.post(
                    api_url,
                    json=request_payload,
                    timeout=timeout
                ) as resp:
                    if resp.status == 200:
                        response_time = time.time() - start_time
                        if self.stream:
                            # Stream response: collect NDJSON chunks
                            response_text = ""
                            async for line in resp.content:
                                if line:
                                    try:
                                        data = json.loads(line.decode('utf-8'))
                                        response_text += data.get("response", "")
                                        if data.get("done"):
                                            break
                                    except json.JSONDecodeError:
                                        continue
                        else:
                            # Non-streaming: single JSON response
                            raw_response_text = await resp.text()
                            logger.debug(f"Raw response from '{model}' (status {resp.status}, length {len(raw_response_text)}): {raw_response_text[:500]}...")
                            try:
                                data = json.loads(raw_response_text)
                                response_text = data.get("response", "")
                            except json.JSONDecodeError as json_e:
                                last_error = f"JSON parsing error from model '{model}': {json_e}. Raw response: {raw_response_text[:500]}"
                                logger.error(last_error)
                                response_text = ""
                        
                        if not response_text.strip():
                            logger.warning(f"Model '{model}' responded successfully but returned an empty or whitespace-only 'response' field.")
                            last_error = f"Model '{model}' returned empty response."
                        else:
                            logger.info(f"Model '{model}' responded successfully in {response_time:.1f}s ({len(response_text)} chars)")
                            result = ModelResponse(
                                model=model,
                                response=response_text,
                                response_time=response_time
                            )
                            if callback:
                                callback(model, "success", result)
                            return result
                    elif resp.status == 404:
                        # Specific handling for 404 - model not found
                        try:
                            error_body = await resp.text()
                        except:
                            error_body = ""
                        last_error = f"HTTP 404: Model '{model}' not found in Ollama. Make sure the model is pulled with 'ollama pull {model}'. Response: {error_body[:200]}"
                        logger.error(last_error)
                    elif resp.status == 500:
                        try:
                            error_body = await resp.text()
                        except:
                            error_body = ""
                        last_error = f"HTTP 500: Ollama server error for model '{model}'. The model may be loading or there's a server issue. Response: {error_body[:200]}"
                        logger.error(last_error)
                    else:
                        try:
                            error_body = await resp.text()
                        except:
                            error_body = ""
                        last_error = f"HTTP {resp.status}: Unexpected status from Ollama for model '{model}'. Response: {error_body[:200]}"
                        logger.error(last_error)
                        
            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.timeout}s waiting for model '{model}' - the model may be too slow or Ollama is overloaded"
                logger.warning(f"Timeout querying '{model}' (attempt {attempt + 1}/{self.retries + 1})")
                if callback and attempt == self.retries:  # Only callback on final timeout
                    callback(model, "timeout", last_error)
            except aiohttp.ClientConnectorError as e:
                last_error = f"Connection failed to {self.ollama_url}: {e}. Is Ollama running? Try 'systemctl status ollama' or 'ollama serve'"
                logger.error(f"Connection error to Ollama: {e}")
            except aiohttp.ClientError as e:
                last_error = f"Network error querying model '{model}': {e}"
                logger.error(f"Client error: {e}")
            except Exception as e:
                last_error = f"Unexpected error querying model '{model}': {type(e).__name__}: {e}"
                logger.exception(f"Unexpected exception querying '{model}'")
            
            # Exponential backoff before retry
            if attempt < self.retries:
                wait_time = 2 ** attempt
                logger.debug(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
        
        # All retries failed
        logger.error(f"All {self.retries + 1} attempts failed for model '{model}': {last_error}")
        if callback:
            callback(model, "error", last_error)
        return ModelResponse(model=model, response="", error=last_error)
    
    async def get_all_responses_async(
        self,
        prompt: str,
        callback: Optional[callable] = None,
        sequential: bool = False
    ) -> List[ModelResponse]:
        """Get responses from all council members (parallel or sequential)"""
        if sequential:
            logger.info(f"Getting responses from {len(self.models)} models sequentially")
        else:
            logger.info(f"Getting responses from {len(self.models)} models in parallel")
        logger.debug(f"Models: {self.models}")
        
        async with aiohttp.ClientSession() as session:
            if sequential:
                responses = []
                for model in self.models:
                    r = await self.query_model_async(session, model, prompt, callback)
                    responses.append(r)
            else:
                tasks = [
                    self.query_model_async(session, model, prompt, callback)
                    for model in self.models
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions, failed responses, and empty responses
            valid_responses = []
            failed_count = 0
            empty_count = 0
            for resp in responses:
                if isinstance(resp, ModelResponse) and not resp.error:
                    # Check for empty responses (no content or only whitespace)
                    if resp.response and resp.response.strip():
                        valid_responses.append(resp)
                        logger.debug(f"Valid response from '{resp.model}' ({len(resp.response)} chars)")
                    else:
                        empty_count += 1
                        logger.warning(f"Model '{resp.model}' returned empty response (length: {len(resp.response) if resp.response else 0})")
                elif isinstance(resp, ModelResponse) and resp.error:
                    failed_count += 1
                    logger.warning(f"Model '{resp.model}' failed: {resp.error}")
                elif isinstance(resp, Exception):
                    failed_count += 1
                    logger.exception(f"Exception during query: {resp}")
            
            logger.info(f"Got {len(valid_responses)} valid responses, {failed_count} failed, {empty_count} empty")
            
            if not valid_responses:
                logger.error(f"No valid responses received from any model! ({failed_count} failed, {empty_count} empty)")
                    
            return valid_responses
    
    def validate_responses(
        self,
        responses: List[ModelResponse],
        original_prompt: str
    ) -> Tuple[List[ModelResponse], List[ModelResponse], Dict[str, dict]]:
        """
        Validate all responses against minimum quality thresholds.
        
        Separates responses into valid and flagged pools. If fewer than 2
        responses pass validation, all responses are returned as valid
        (best-of-a-bad-lot fallback).
        
        Args:
            responses: List of ModelResponse objects to validate
            original_prompt: The original user prompt for relevance checking
            
        Returns:
            Tuple of (valid_responses, flagged_responses, validation_details)
            - valid_responses: responses that passed validation (or all, if fallback)
            - flagged_responses: responses that failed validation
            - validation_details: dict mapping model name to validation result
        """
        valid = []
        flagged = []
        details = {}
        
        for resp in responses:
            result = ResponseValidator.validate_response(resp.response, original_prompt)
            details[resp.model] = result
            
            if result['is_valid']:
                valid.append(resp)
                logger.debug(
                    f"Response from '{resp.model}' passed validation "
                    f"(relevance: {result['relevance_score']:.2f})"
                )
            else:
                flagged.append(resp)
                logger.warning(
                    f"Response from '{resp.model}' FAILED validation: "
                    f"{'; '.join(result['reasons'])}"
                )
        
        logger.info(
            f"Validation complete: {len(valid)} valid, {len(flagged)} flagged "
            f"out of {len(responses)} total"
        )
        
        # Fallback: if fewer than 2 valid responses, use all responses
        if len(valid) < 2 and responses:
            logger.warning(
                f"Only {len(valid)} valid response(s) — falling back to all "
                f"{len(responses)} responses for voting"
            )
            return responses, flagged, details
        
        return valid, flagged, details
    
    def create_voting_prompt(
        self,
        original_prompt: str,
        responses: List[ModelResponse],
        exclude_model: Optional[str] = None
    ) -> Tuple[str, List[ModelResponse]]:
        """
        Create anonymized voting prompt.
        
        Returns the prompt and the filtered responses list (for label mapping).
        """
        # Filter out the voter's own response if exclude_self_vote is enabled
        filtered_responses = responses
        if exclude_model and self.exclude_self_vote:
            filtered_responses = [r for r in responses if r.model != exclude_model]
        
        if not filtered_responses:
            return "", []
        
        voting_prompt = f"""You are evaluating responses to this question:

"{original_prompt}"

Here are the candidate responses:

"""
        for i, resp in enumerate(filtered_responses):
            if i < len(self.LABELS):
                voting_prompt += f"=== Response {self.LABELS[i]} ===\n{resp.response}\n\n"
        
        num_responses = len(filtered_responses)
        example_ranking = ' '.join(self.LABELS[:num_responses])
        
        voting_prompt += f"""
Evaluate each response for:
1. Accuracy and correctness
2. Completeness
3. Clarity and organization
4. Relevance to the question

CRITICAL: If any response is a refusal (e.g. "I'm sorry, I can't assist"), off-topic, or does not meaningfully address the question, it MUST be ranked last regardless of other qualities.

Rank ALL {num_responses} responses from best to worst.

IMPORTANT: Reply with ONLY a JSON object in this exact format:
{{"ranking": ["{self.LABELS[0]}", "{self.LABELS[1]}", "{self.LABELS[2] if num_responses > 2 else self.LABELS[1]}"]}}

Your ranking (JSON only):"""
        
        return voting_prompt, filtered_responses
    
    def parse_ranking(self, ranking_text: str, num_responses: int) -> List[str]:
        """
        Parse ranking from model response with multiple fallback strategies.
        """
        valid_labels = set(self.LABELS[:num_responses])
        
        # Strategy 1: Try JSON parsing
        try:
            # Find JSON object in response
            json_match = re.search(r'\{[^}]+\}', ranking_text)
            if json_match:
                data = json.loads(json_match.group())
                if 'ranking' in data and isinstance(data['ranking'], list):
                    ranking = [l.upper() for l in data['ranking'] if l.upper() in valid_labels]
                    if ranking:
                        return ranking
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Strategy 2: Extract letters in order
        letters = []
        for char in ranking_text.upper():
            if char in valid_labels and char not in letters:
                letters.append(char)
        
        if letters:
            return letters
        
        # Strategy 3: Look for patterns like "A > B > C" or "A, B, C" or "1. A 2. B"
        patterns = [
            r'([A-J])\s*[>→,]\s*([A-J])\s*[>→,]\s*([A-J])',  # A > B > C
            r'1[.):]\s*([A-J]).*2[.):]\s*([A-J]).*3[.):]\s*([A-J])',  # 1. A 2. B 3. C
        ]
        for pattern in patterns:
            match = re.search(pattern, ranking_text.upper())
            if match:
                ranking = [g for g in match.groups() if g in valid_labels]
                if ranking:
                    return ranking
        
        # Fallback: return first valid label found
        for char in ranking_text.upper():
            if char in valid_labels:
                return [char]
        
        return []
    
    async def conduct_vote_async(
        self,
        original_prompt: str,
        responses: List[ModelResponse],
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Have models vote on responses asynchronously"""
        votes = []
        
        async with aiohttp.ClientSession() as session:
            for model in self.models:
                voting_prompt, filtered_responses = self.create_voting_prompt(
                    original_prompt, responses, exclude_model=model
                )
                
                if not filtered_responses:
                    continue
                
                if callback:
                    callback(model, "voting", None)
                
                vote_response = await self.query_model_async(
                    session, model, voting_prompt, callback=None
                )
                
                if not vote_response.error:
                    ranking = self.parse_ranking(vote_response.response, len(filtered_responses))
                    
                    # Map back to original response indices
                    vote_data = {
                        "voter": model,
                        "ranking": ranking,
                        "raw_response": vote_response.response,
                        "voted_on": [r.model for r in filtered_responses]
                    }
                    votes.append(vote_data)
                    
                    if callback:
                        callback(model, "voted", ranking)
        
        return self.tally_votes(votes, responses)
    
    def tally_votes(
        self,
        votes: List[Dict[str, Any]],
        responses: List[ModelResponse]
    ) -> Dict[str, Any]:
        """Tally votes and determine winner using Borda count"""
        labels = self.LABELS[:len(responses)]
        scores = {label: 0 for label in labels}
        
        # Points decrease by position: 1st = n points, 2nd = n-1, etc.
        max_points = len(responses)
        
        for vote in votes:
            ranking = vote.get("ranking", [])
            for rank, label in enumerate(ranking):
                if label in scores:
                    points = max(max_points - rank, 1)
                    scores[label] += points
        
        # Find winner
        if not scores or all(v == 0 for v in scores.values()):
            # No valid votes, return first response
            winner_label = labels[0] if labels else 'A'
            winner_idx = 0
        else:
            winner_label = max(scores, key=scores.get)
            winner_idx = labels.index(winner_label)
        
        return {
            "scores": scores,
            "winner": responses[winner_idx] if winner_idx < len(responses) else responses[0],
            "winner_label": winner_label,
            "all_votes": votes
        }
    
    async def synthesize_response_async(
        self,
        original_prompt: str,
        responses: List[ModelResponse],
        top_n: int = 3
    ) -> ModelResponse:
        """
        Synthesis mode: Create a new response combining best elements.
        Uses the top-ranked model to synthesize.
        """
        synthesis_prompt = f"""Original question: {original_prompt}

Here are several responses to this question:

"""
        for i, resp in enumerate(responses[:top_n]):
            synthesis_prompt += f"Response {i+1} (from {resp.model}):\n{resp.response}\n\n"
        
        synthesis_prompt += """
Create a comprehensive response that:
1. Combines the best insights from all responses above
2. Resolves any contradictions
3. Fills in any gaps
4. Is well-organized and clear

Synthesized response:"""
        
        async with aiohttp.ClientSession() as session:
            # Use the first model for synthesis
            return await self.query_model_async(session, self.models[0], synthesis_prompt)
    
    async def debate_and_refine_async(
        self,
        original_prompt: str,
        winner: ModelResponse,
        all_responses: List[ModelResponse]
    ) -> ModelResponse:
        """
        Debate mode: Have models critique the winner and refine it.
        """
        critique_prompt = f"""Original question: {original_prompt}

Winning response:
{winner.response}

As a critical reviewer, identify:
1. Any errors or inaccuracies
2. Missing important information
3. Areas that could be clearer

Then provide an improved version that addresses these issues.

Critique and improved response:"""
        
        async with aiohttp.ClientSession() as session:
            # Use a different model than the winner for critique
            critic_model = self.models[0] if winner.model != self.models[0] else self.models[-1]
            return await self.query_model_async(session, critic_model, critique_prompt)
    
    async def run_council_async(
        self,
        prompt: str,
        mode: CouncilMode = CouncilMode.RANKED,
        callback: Optional[callable] = None
    ) -> VoteResult:
        """
        Full council workflow (async version).
        
        Args:
            prompt: The question/prompt to send to all models
            mode: Council voting mode
            callback: Optional callback for progress updates
                     signature: callback(model: str, stage: str, data: Any)
        """
        if callback:
            callback(None, "start", {"prompt": prompt, "models": self.models})
        
        # Stage 1: Get all responses in parallel
        if callback:
            callback(None, "gathering", None)
        
        responses = await self.get_all_responses_async(prompt, callback)
        
        if not responses:
            raise ValueError("No valid responses received from any model")
        
        # Stage 2: Vote on responses
        if callback:
            callback(None, "voting_start", None)
        
        vote_results = await self.conduct_vote_async(prompt, responses, callback)
        
        # Stage 3: Apply mode-specific post-processing
        winner = vote_results["winner"]
        
        if mode == CouncilMode.SYNTHESIS:
            if callback:
                callback(None, "synthesizing", None)
            winner = await self.synthesize_response_async(prompt, responses)
        
        elif mode == CouncilMode.DEBATE:
            if callback:
                callback(None, "debating", None)
            winner = await self.debate_and_refine_async(prompt, winner, responses)
        
        result = VoteResult(
            scores=vote_results["scores"],
            winner=winner,
            winner_label=vote_results["winner_label"],
            all_responses=responses,
            all_votes=vote_results["all_votes"],
            mode=mode
        )
        
        if callback:
            callback(None, "complete", result)
        
        return result
    
    def run_council(
        self,
        prompt: str,
        mode: CouncilMode = CouncilMode.RANKED,
        verbose: bool = True
    ) -> VoteResult:
        """
        Synchronous wrapper for run_council_async.
        
        Args:
            prompt: The question/prompt to send to all models
            mode: Council voting mode (RANKED, MAJORITY, SYNTHESIS, DEBATE)
            verbose: Whether to print progress
        """
        def print_callback(model, stage, data):
            if not verbose:
                return
            if stage == "start":
                print(f"\n{'='*60}")
                print(f"COUNCIL QUERY: {data['prompt'][:50]}...")
                print(f"Models: {', '.join(data['models'])}")
                print(f"{'='*60}\n")
            elif stage == "gathering":
                print("Stage 1: Gathering responses from council members...\n")
            elif stage == "success":
                print(f"  {model}: Response received ({data.response_time:.1f}s)")
            elif stage == "error":
                print(f"  {model}: Error - {data}")
            elif stage == "voting_start":
                print(f"\nStage 2: Council voting (mode: {mode.value})...\n")
            elif stage == "voted":
                if data:
                    print(f"  {model}: Ranked {' > '.join(data)}")
            elif stage == "synthesizing":
                print("\nStage 3: Synthesizing combined response...")
            elif stage == "debating":
                print("\nStage 3: Debating and refining winner...")
            elif stage == "complete":
                print(f"\nResults:")
                print(f"  Scores: {data.scores}")
                print(f"  Winner: {data.winner.model} (Response {data.winner_label})")
        
        return asyncio.run(self.run_council_async(prompt, mode, print_callback if verbose else None))
    
    # Legacy synchronous methods for backwards compatibility
    def query_model(self, model: str, prompt: str) -> str:
        """Synchronous single model query (legacy compatibility)"""
        import requests
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=self.timeout
        )
        return response.json()["response"]
    
    def get_all_responses(self, prompt: str) -> List[Dict[str, str]]:
        """Synchronous response gathering (legacy compatibility)"""
        responses = asyncio.run(self.get_all_responses_async(prompt))
        return [{"model": r.model, "response": r.response} for r in responses]


# Example usage
if __name__ == "__main__":
    # Initialize council with your models
    council = LLMCouncil(
        models=["phi3:mini", "tinyllama", "qwen2.5:0.5b"],
        timeout=600,  # 10 minutes - extended timeout for slow models on limited RAM
        retries=2,
        exclude_self_vote=True  # Prevent models from voting on their own responses
    )
    
    # Run with default ranked voting
    results = council.run_council(
        "Explain the difference between async and multithreading in Python",
        mode=CouncilMode.RANKED
    )
    
    print(f"\n{'='*60}")
    print("WINNING RESPONSE:")
    print(f"{'='*60}")
    print(results.winner.response)
