#!/usr/bin/env python3
"""
Creative Writing Churn Engine

Iterative creative writing system that uses the council to provide
synthesized feedback, diffs, and expanded content for creative writing projects.

Features:
- Three churn modes: EDIT, EXPAND, EXPLORE
- Synthesized feedback from multiple models
- Unified diff generation
- Iteration tracking and branching support
"""

import asyncio
import aiohttp
import difflib
import json
import logging
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

# Import the base council for model querying
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from council import LLMCouncil, ModelResponse

# Set up logger for debugging
logger = logging.getLogger('council_app.churn')


class ChurnType(Enum):
    """Types of churn operations for creative writing"""
    EDIT = "edit"           # Line-editing with feedback
    EXPAND = "expand"       # Expand outline to prose
    EXPLORE = "explore"     # Generate variations
    REPORT = "report"       # Technical report section review


@dataclass
class ChurnResult:
    """Result of a churn operation"""
    original_content: str
    suggested_content: str
    synthesized_feedback: str
    content_diff: str
    raw_responses: List[Dict[str, Any]]
    churn_type: ChurnType
    processing_time: float


class ChurnEngine:
    """
    Creative writing iteration engine using council feedback.
    
    Focuses on thorough exploration rather than speed - each iteration
    gathers feedback from multiple models and synthesizes it into
    actionable suggestions.
    """
    
    def __init__(
        self,
        models: List[str],
        ollama_url: str = "http://localhost:11434",
        timeout: int = 120,  # Longer timeout for creative work
        retries: int = 2
    ):
        self.models = models
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.retries = retries
        # Use LLMCouncil for model querying
        self.council = LLMCouncil(
            models=models,
            ollama_url=ollama_url,
            timeout=timeout,
            retries=retries,
            exclude_self_vote=False  # Not voting, so no need to exclude
        )
    
    # =========================================================================
    # PROMPT TEMPLATES
    # =========================================================================
    
    def create_edit_prompt(self, content: str, context: str = "") -> str:
        """
        Create a prompt for editing/refining creative writing.
        
        The edit prompt asks models to act as editors, providing:
        - Line-by-line suggestions
        - Grammar and style corrections
        - Structural improvements
        - A revised version
        """
        context_section = f"\nContext/Genre: {context}\n" if context else ""
        
        return f"""You are an expert creative writing editor. Review the following piece of writing and provide detailed editorial feedback.
{context_section}
=== WRITING TO EDIT ===
{content}
=== END OF WRITING ===

Please provide:

1. **Overall Assessment** (2-3 sentences): What works well and what needs improvement?

2. **Specific Suggestions**: List 3-5 concrete improvements with line references where applicable:
   - Grammar/punctuation issues
   - Awkward phrasing or unclear passages
   - Pacing or structural issues
   - Opportunities for stronger word choices
   - Character/voice consistency (if applicable)

3. **Revised Version**: Provide a complete revised version of the writing that incorporates your suggestions. Keep the author's voice and intent while improving clarity, flow, and impact.

Format your response as:
ASSESSMENT:
[Your overall assessment]

SUGGESTIONS:
1. [Suggestion 1]
2. [Suggestion 2]
...

REVISED VERSION:
[Complete revised text]"""

    def create_expand_prompt(self, outline: str, context: str = "", style_notes: str = "") -> str:
        """
        Create a prompt for expanding an outline into prose.
        
        The expand prompt asks models to:
        - Develop scene descriptions
        - Add dialogue
        - Flesh out characters
        - Maintain narrative voice
        """
        context_section = f"\nGenre/Context: {context}" if context else ""
        style_section = f"\nStyle Notes: {style_notes}" if style_notes else ""
        
        return f"""You are a skilled creative writer. Your task is to expand the following outline into fully developed prose.
{context_section}{style_section}

=== OUTLINE TO EXPAND ===
{outline}
=== END OF OUTLINE ===

Guidelines for expansion:
1. **Show, Don't Tell**: Use sensory details, actions, and dialogue instead of direct exposition
2. **Develop Characters**: Give characters distinct voices, mannerisms, and motivations
3. **Set the Scene**: Include atmospheric details that establish mood and place
4. **Pacing**: Vary sentence length and paragraph structure for rhythm
5. **Dialogue**: Make conversations feel natural with subtext and character voice
6. **Maintain Coherence**: Ensure smooth transitions between scenes/sections

Please expand this outline into engaging prose. Take your time - quality and depth matter more than brevity.

EXPANDED VERSION:"""

    def create_explore_prompt(
        self, 
        content: str, 
        direction: Optional[str] = None,
        context: str = ""
    ) -> str:
        """
        Create a prompt for exploring variations of the writing.
        
        The explore prompt asks models to generate:
        - Alternative plot directions
        - Different tones/styles
        - "What if" scenarios
        - Multiple options to branch from
        """
        context_section = f"\nContext/Genre: {context}" if context else ""
        
        direction_instruction = ""
        if direction:
            direction_instruction = f"""
Exploration Direction: {direction}
Focus your variations on this specific aspect."""
        else:
            direction_instruction = """
Generate variations that explore different possibilities:
- Alternative narrative directions
- Different tonal approaches
- Structural reorganizations
- Character perspective shifts"""

        return f"""You are a creative writing consultant helping explore different directions for a piece of writing.
{context_section}

=== ORIGINAL WRITING ===
{content}
=== END OF ORIGINAL ===
{direction_instruction}

Please provide 2-3 distinct variations or alternative approaches. For each variation:

1. **Variation Title**: A brief descriptor (e.g., "Darker Tone" or "From the Antagonist's View")
2. **Key Changes**: What's different about this approach (2-3 bullet points)
3. **Sample**: A substantial excerpt showing this variation in action

Format your response as:

--- VARIATION 1: [Title] ---
Key Changes:
- [Change 1]
- [Change 2]

Sample:
[Extended writing sample]

--- VARIATION 2: [Title] ---
...

Be creative and bold - the goal is to explore possibilities, not play it safe."""

    def create_synthesis_prompt(self, responses: List[Dict[str, Any]], churn_type: ChurnType) -> str:
        """
        Create a prompt for synthesizing multiple model responses into unified feedback.
        """
        responses_text = ""
        for i, resp in enumerate(responses, 1):
            responses_text += f"\n=== RESPONSE {i} (from {resp['model']}) ===\n{resp['response']}\n"
        
        if churn_type == ChurnType.EDIT:
            synthesis_instruction = """Synthesize these editorial responses into a single, comprehensive feedback document:
1. Combine overlapping suggestions (note when multiple editors agreed)
2. Resolve contradictory advice by explaining the trade-offs
3. Prioritize suggestions by impact
4. Create ONE definitive revised version that incorporates the best changes"""
        
        elif churn_type == ChurnType.EXPAND:
            synthesis_instruction = """Synthesize these expanded versions into the best possible prose:
1. Identify the strongest passages from each version
2. Combine the best elements while maintaining consistency
3. Ensure smooth flow and transitions
4. Produce ONE cohesive expanded version"""
        
        else:  # EXPLORE
            synthesis_instruction = """Synthesize these explorations into a curated collection:
1. Identify the most promising and distinct variations
2. Remove redundant or weak options
3. Note which variations might work well together
4. Present the top 2-3 variations with clear descriptions"""

        return f"""You are synthesizing feedback from multiple creative writing consultants.
{responses_text}

{synthesis_instruction}

SYNTHESIZED FEEDBACK:
[Your synthesis here]

FINAL SUGGESTED VERSION:
[The combined/best version]"""

    # =========================================================================
    # DIFF GENERATION
    # =========================================================================
    
    def generate_unified_diff(
        self, 
        original: str, 
        revised: str, 
        context_lines: int = 3
    ) -> str:
        """
        Generate a unified diff between original and revised content.
        
        Returns a human-readable diff with context lines.
        """
        original_lines = original.splitlines(keepends=True)
        revised_lines = revised.splitlines(keepends=True)
        
        # Ensure lines end with newline for clean diff
        if original_lines and not original_lines[-1].endswith('\n'):
            original_lines[-1] += '\n'
        if revised_lines and not revised_lines[-1].endswith('\n'):
            revised_lines[-1] += '\n'
        
        diff = difflib.unified_diff(
            original_lines,
            revised_lines,
            fromfile='Original',
            tofile='Revised',
            n=context_lines
        )
        
        return ''.join(diff)
    
    def generate_html_diff(self, original: str, revised: str) -> str:
        """
        Generate an HTML diff for display in the web interface.
        """
        differ = difflib.HtmlDiff(wrapcolumn=80)
        return differ.make_table(
            original.splitlines(),
            revised.splitlines(),
            fromdesc='Original',
            todesc='Revised',
            context=True,
            numlines=3
        )
    
    def generate_word_diff(self, original: str, revised: str) -> List[Dict[str, Any]]:
        """
        Generate a word-level diff for fine-grained visualization.
        
        Returns a list of diff operations: {'type': 'equal'|'insert'|'delete', 'text': str}
        """
        # Split into words while preserving whitespace
        def tokenize(text):
            return re.findall(r'\S+|\s+', text)
        
        original_words = tokenize(original)
        revised_words = tokenize(revised)
        
        matcher = difflib.SequenceMatcher(None, original_words, revised_words)
        
        result = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                result.append({
                    'type': 'equal',
                    'text': ''.join(original_words[i1:i2])
                })
            elif tag == 'delete':
                result.append({
                    'type': 'delete',
                    'text': ''.join(original_words[i1:i2])
                })
            elif tag == 'insert':
                result.append({
                    'type': 'insert',
                    'text': ''.join(revised_words[j1:j2])
                })
            elif tag == 'replace':
                result.append({
                    'type': 'delete',
                    'text': ''.join(original_words[i1:i2])
                })
                result.append({
                    'type': 'insert',
                    'text': ''.join(revised_words[j1:j2])
                })
        
        return result

    # =========================================================================
    # RESPONSE PARSING
    # =========================================================================
    
    def parse_edit_response(self, response: str) -> Dict[str, Any]:
        """Parse an edit response to extract assessment, suggestions, and revised version."""
        result = {
            'assessment': '',
            'suggestions': [],
            'revised_version': ''
        }
        
        # Try to extract sections
        assessment_match = re.search(
            r'ASSESSMENT:\s*(.*?)(?=SUGGESTIONS:|REVISED VERSION:|$)', 
            response, 
            re.DOTALL | re.IGNORECASE
        )
        if assessment_match:
            result['assessment'] = assessment_match.group(1).strip()
        
        suggestions_match = re.search(
            r'SUGGESTIONS:\s*(.*?)(?=REVISED VERSION:|$)', 
            response, 
            re.DOTALL | re.IGNORECASE
        )
        if suggestions_match:
            suggestions_text = suggestions_match.group(1).strip()
            # Parse numbered suggestions
            result['suggestions'] = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', suggestions_text, re.DOTALL)
            result['suggestions'] = [s.strip() for s in result['suggestions'] if s.strip()]
        
        revised_match = re.search(
            r'REVISED VERSION:\s*(.*?)$', 
            response, 
            re.DOTALL | re.IGNORECASE
        )
        if revised_match:
            result['revised_version'] = revised_match.group(1).strip()
        
        # If no structure found, use the whole response as revised version
        if not result['revised_version'] and not result['assessment']:
            result['revised_version'] = response.strip()
        
        return result
    
    def parse_expand_response(self, response: str) -> str:
        """Parse an expand response to extract the expanded content."""
        # Try to find content after "EXPANDED VERSION:" marker
        match = re.search(r'EXPANDED VERSION:\s*(.*?)$', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Otherwise return the whole response
        return response.strip()
    
    def parse_explore_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse an explore response to extract variations."""
        variations = []
        
        # Find all variation sections
        variation_pattern = r'---\s*VARIATION\s*\d+:\s*(.+?)\s*---\s*(.*?)(?=---\s*VARIATION|$)'
        matches = re.findall(variation_pattern, response, re.DOTALL | re.IGNORECASE)
        
        for title, content in matches:
            variation = {
                'title': title.strip(),
                'changes': [],
                'sample': ''
            }
            
            # Extract key changes
            changes_match = re.search(r'Key Changes:\s*(.*?)(?=Sample:|$)', content, re.DOTALL | re.IGNORECASE)
            if changes_match:
                changes_text = changes_match.group(1)
                variation['changes'] = re.findall(r'-\s*(.+?)(?=-|$)', changes_text)
                variation['changes'] = [c.strip() for c in variation['changes'] if c.strip()]
            
            # Extract sample
            sample_match = re.search(r'Sample:\s*(.*?)$', content, re.DOTALL | re.IGNORECASE)
            if sample_match:
                variation['sample'] = sample_match.group(1).strip()
            
            if variation['sample'] or variation['changes']:
                variations.append(variation)
        
        # If no structured variations found, treat whole response as one variation
        if not variations:
            variations.append({
                'title': 'Variation',
                'changes': [],
                'sample': response.strip()
            })
        
        return variations

    def parse_synthesis_response(self, response: str) -> Dict[str, str]:
        """Parse a synthesis response to extract feedback and suggested content."""
        result = {
            'feedback': '',
            'suggested_content': ''
        }
        
        # Try to extract sections
        feedback_match = re.search(
            r'SYNTHESIZED FEEDBACK:\s*(.*?)(?=FINAL SUGGESTED VERSION:|$)', 
            response, 
            re.DOTALL | re.IGNORECASE
        )
        if feedback_match:
            result['feedback'] = feedback_match.group(1).strip()
        
        suggested_match = re.search(
            r'FINAL SUGGESTED VERSION:\s*(.*?)$', 
            response, 
            re.DOTALL | re.IGNORECASE
        )
        if suggested_match:
            result['suggested_content'] = suggested_match.group(1).strip()
        
        # Fallback
        if not result['feedback']:
            result['feedback'] = response.strip()
        
        return result

    # =========================================================================
    # MAIN PROCESSING METHODS
    # =========================================================================
    
    async def process_iteration_async(
        self,
        content: str,
        churn_type: ChurnType,
        context: str = "",
        direction: Optional[str] = None,
        callback: Optional[callable] = None
    ) -> ChurnResult:
        """
        Main async method for processing an iteration.
        
        Gathers feedback from all models, synthesizes it, and generates diffs.
        """
        start_time = time.time()
        logger.info(f"=== Starting churn processing (type: {churn_type.value}) ===")
        logger.debug(f"Content length: {len(content)} chars, Context: '{context}', Direction: '{direction}'")
        logger.debug(f"Models to query: {self.models}")
        logger.debug(f"Ollama URL: {self.ollama_url}, Timeout: {self.timeout}s")
        
        # Stage 1: Create the appropriate prompt
        logger.info("Stage 1: Creating prompt")
        if callback:
            callback(None, "creating_prompt", churn_type)
        
        if churn_type == ChurnType.EDIT:
            prompt = self.create_edit_prompt(content, context)
        elif churn_type == ChurnType.EXPAND:
            prompt = self.create_expand_prompt(content, context)
        else:  # EXPLORE
            prompt = self.create_explore_prompt(content, direction, context)
        
        logger.debug(f"Created prompt ({len(prompt)} chars)")
        
        # Stage 2: Gather responses from all models
        logger.info("Stage 2: Gathering responses from all models")
        if callback:
            callback(None, "gathering", None)
        
        responses = await self.council.get_all_responses_async(prompt, callback)
        
        logger.info(f"Received {len(responses) if responses else 0} valid responses")
        for r in (responses or []):
            logger.debug(f"  - {r.model}: {'OK' if not r.error else f'ERROR: {r.error}'} ({r.response_time:.1f}s)")
        
        if not responses:
            logger.error("No valid responses received from any model!")
            raise ValueError("No valid responses received from any model")
        
        # Convert to dict format for storage
        raw_responses = [
            {
                'model': r.model,
                'response': r.response,
                'response_time': r.response_time
            }
            for r in responses
        ]
        
        # Stage 3: Synthesize responses
        logger.info("Stage 3: Synthesizing responses")
        if callback:
            callback(None, "synthesizing", None)
        
        synthesis_prompt = self.create_synthesis_prompt(raw_responses, churn_type)
        logger.debug(f"Synthesis prompt length: {len(synthesis_prompt)} chars")
        
        logger.debug(f"Querying {self.models[0]} for synthesis")
        async with aiohttp.ClientSession() as session:
            synthesis_response = await self.council.query_model_async(
                session, 
                self.models[0],  # Use first model for synthesis
                synthesis_prompt
            )
        
        if synthesis_response.error:
            logger.error(f"Synthesis failed with error: {synthesis_response.error}")
            raise ValueError(f"Synthesis failed: {synthesis_response.error}")
        
        logger.debug(f"Synthesis response received ({len(synthesis_response.response)} chars)")
        
        # Stage 4: Parse synthesis and extract suggested content
        logger.info("Stage 4: Parsing synthesis response")
        if callback:
            callback(None, "parsing", None)
        
        parsed_synthesis = self.parse_synthesis_response(synthesis_response.response)
        synthesized_feedback = parsed_synthesis['feedback']
        suggested_content = parsed_synthesis['suggested_content']
        
        logger.debug(f"Parsed feedback: {len(synthesized_feedback)} chars")
        logger.debug(f"Parsed suggested content: {len(suggested_content)} chars")
        
        # If no suggested content from synthesis, try to extract from individual responses
        if not suggested_content:
            logger.warning("No suggested content from synthesis, extracting from individual responses")
            if churn_type == ChurnType.EDIT:
                # Try to get revised version from first response
                parsed = self.parse_edit_response(responses[0].response)
                suggested_content = parsed.get('revised_version', content)
            elif churn_type == ChurnType.EXPAND:
                suggested_content = self.parse_expand_response(responses[0].response)
            else:  # EXPLORE
                variations = self.parse_explore_response(responses[0].response)
                if variations:
                    suggested_content = variations[0].get('sample', content)
            logger.debug(f"Extracted suggested content: {len(suggested_content)} chars")
        
        # Stage 5: Generate diff
        logger.info("Stage 5: Generating diff")
        if callback:
            callback(None, "generating_diff", None)
        
        content_diff = self.generate_unified_diff(content, suggested_content)
        logger.debug(f"Generated diff: {len(content_diff)} chars")
        
        processing_time = time.time() - start_time
        
        if callback:
            callback(None, "complete", None)
        
        logger.info(f"=== Churn processing completed in {processing_time:.1f}s ===")
        
        return ChurnResult(
            original_content=content,
            suggested_content=suggested_content,
            synthesized_feedback=synthesized_feedback,
            content_diff=content_diff,
            raw_responses=raw_responses,
            churn_type=churn_type,
            processing_time=processing_time
        )
    
    def process_iteration(
        self,
        content: str,
        churn_type: ChurnType,
        context: str = "",
        direction: Optional[str] = None,
        verbose: bool = True
    ) -> ChurnResult:
        """
        Synchronous wrapper for process_iteration_async.
        """
        def print_callback(model, stage, data):
            if not verbose:
                return
            if stage == "creating_prompt":
                print(f"\n{'='*60}")
                print(f"CHURN MODE: {data.value.upper()}")
                print(f"Models: {', '.join(self.models)}")
                print(f"{'='*60}\n")
            elif stage == "gathering":
                print("Stage 1: Gathering feedback from council members...\n")
            elif stage == "success":
                print(f"  {model}: Response received")
            elif stage == "error":
                print(f"  {model}: Error - {data}")
            elif stage == "synthesizing":
                print("\nStage 2: Synthesizing feedback...")
            elif stage == "parsing":
                print("Stage 3: Parsing responses...")
            elif stage == "generating_diff":
                print("Stage 4: Generating diff...")
            elif stage == "complete":
                print("\nProcessing complete!")
        
        return asyncio.run(
            self.process_iteration_async(
                content, churn_type, context, direction,
                print_callback if verbose else None
            )
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def apply_diff(original: str, diff_text: str) -> str:
    """
    Apply a unified diff to original content.
    
    Note: This is a simple implementation. For production, consider using
    the `patch` library or similar.
    """
    # For now, we just return the diff as-is since we store the suggested content
    # This function could be expanded to actually apply diffs
    return original


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two texts.
    Returns a float between 0 and 1.
    """
    return difflib.SequenceMatcher(None, text1, text2).ratio()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    engine = ChurnEngine(
        models=["phi3:mini", "tinyllama"],
        timeout=120
    )
    
    sample_content = """
    The old house stood at the end of the lane. It was dark. 
    Sarah walked towards it. She was scared but also curious.
    The door was open. She went inside.
    """
    
    print("Testing EDIT mode...")
    result = engine.process_iteration(
        content=sample_content,
        churn_type=ChurnType.EDIT,
        context="Horror short story"
    )
    
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"\nProcessing time: {result.processing_time:.1f}s")
    print(f"\nSynthesized Feedback:\n{result.synthesized_feedback[:500]}...")
    print(f"\nDiff:\n{result.content_diff[:500]}...")
