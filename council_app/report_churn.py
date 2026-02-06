#!/usr/bin/env python3
"""
Technical Report Churn Engine

Section-by-section report review system that uses the council to provide
compliance-oriented feedback against a knowledgebase.

Features:
- Outline parsing (markdown headers and numbered lists)
- Per-section council review with knowledgebase context
- Compliance scoring
- Synthesized feedback from multiple models
- Diff generation between original and revised sections
"""

import asyncio
import aiohttp
import difflib
import json
import logging
import re
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# Import the base council for model querying
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from council import LLMCouncil, ModelResponse

# Set up logger for debugging
logger = logging.getLogger('council_app.report_churn')


# =============================================================================
# OUTLINE PARSER
# =============================================================================

class OutlineParser:
    """
    Parses markdown outlines into structured sections.

    Supports formats like:
        # Main Section
        Content here...

        ## Subsection
        More content...

    Or numbered:
        1. First Section
           1.1 Subsection
    """

    def parse(self, raw_outline: str) -> List[Dict]:
        """Parse outline into list of section dicts."""
        sections = []
        current_section = None
        section_counter = 0
        content_buffer = []

        lines = raw_outline.split('\n')

        for line in lines:
            # Check for markdown header patterns
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            # Check for numbered patterns like "1.", "1.1", "1.2.3)"
            numbered_match = re.match(r'^(\d+(?:\.\d+)*)[.)\s]+(.+)$', line)

            if header_match or numbered_match:
                # Save previous section
                if current_section:
                    current_section['content'] = '\n'.join(content_buffer).strip()
                    sections.append(current_section)
                    content_buffer = []

                section_counter += 1

                if header_match:
                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()
                else:
                    # Count dots for level (1.2.3 = level 3)
                    level = numbered_match.group(1).count('.') + 1
                    title = numbered_match.group(2).strip()

                current_section = {
                    'id': f"section_{section_counter}",
                    'title': title,
                    'content': '',
                    'level': level,
                    'parent_id': self._find_parent(sections, level),
                    'order': section_counter
                }
            else:
                content_buffer.append(line)

        # Don't forget last section
        if current_section:
            current_section['content'] = '\n'.join(content_buffer).strip()
            sections.append(current_section)

        return sections

    def _find_parent(self, sections: List[Dict], current_level: int) -> Optional[str]:
        """Find the most recent section with a lower level (parent)."""
        for section in reversed(sections):
            if section['level'] < current_level:
                return section['id']
        return None


# =============================================================================
# REPORT SECTION REVIEW RESULT
# =============================================================================

@dataclass
class SectionReviewResult:
    """Result of reviewing a single report section."""
    section_id: str
    section_title: str
    original_content: str
    revised_content: str
    synthesized_feedback: str
    content_diff: str
    compliance_score: float
    raw_responses: List[Dict[str, Any]]
    processing_time: float


# =============================================================================
# REPORT CHURN ENGINE
# =============================================================================

class ReportChurnEngine:
    """
    Technical report section review engine using council feedback.

    Processes report sections one at a time, using knowledgebase context
    and preceding section context to provide compliance-oriented feedback.
    """

    def __init__(
        self,
        models: List[str],
        ollama_url: str = "http://localhost:11434",
        timeout: int = 120,
        retries: int = 2,
        knowledgebase_content: str = ""
    ):
        self.models = models
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.retries = retries
        self.knowledgebase_content = knowledgebase_content
        # Use LLMCouncil for model querying (same pattern as ChurnEngine)
        self.council = LLMCouncil(
            models=models,
            ollama_url=ollama_url,
            timeout=timeout,
            retries=retries,
            exclude_self_vote=False
        )

    # =========================================================================
    # PROMPT TEMPLATES
    # =========================================================================

    def create_section_review_prompt(
        self,
        section_title: str,
        section_content: str,
        knowledge_context: str,
        full_outline: str,
        preceding_sections: Optional[List[Dict]] = None
    ) -> str:
        """
        Create a prompt for reviewing a single report section.

        Includes knowledgebase guidelines, the full outline for context,
        preceding approved sections, and the current section to review.
        """
        kb_section = ""
        if knowledge_context:
            kb_section = f"""
=== KNOWLEDGEBASE GUIDELINES ===
{knowledge_context}
=== END GUIDELINES ===
"""

        preceding_context = ""
        if preceding_sections:
            preceding_context = "\n=== PRECEDING SECTIONS (for context) ===\n"
            for sec in preceding_sections[-3:]:  # Last 3 sections for context window
                content_preview = sec['content'][:500]
                if len(sec['content']) > 500:
                    content_preview += "..."
                preceding_context += f"\n## {sec['title']}\n{content_preview}\n"
            preceding_context += "=== END PRECEDING SECTIONS ===\n"

        return f"""You are a technical report reviewer with expertise in structured documentation.
{kb_section}
=== FULL REPORT OUTLINE ===
{full_outline}
=== END OUTLINE ===
{preceding_context}
=== CURRENT SECTION TO REVIEW ===
Section: {section_title}
Content:
{section_content}
=== END SECTION ===

Review this section against the knowledgebase guidelines (if provided) and the overall report structure. Provide:

1. **COMPLIANCE ASSESSMENT** (score 0-10):
   - Does this section follow the structural guidelines?
   - Are required elements present?
   - Is the level of detail appropriate?

2. **CONTENT QUALITY**:
   - Clarity and precision of writing
   - Technical accuracy considerations
   - Flow from preceding sections

3. **SPECIFIC IMPROVEMENTS**:
   - List 3-5 concrete changes
   - Reference specific guidelines from the knowledgebase where applicable

4. **REVISED SECTION**:
   - Provide an improved version incorporating your feedback
   - Maintain the author's voice while improving structure and compliance

Format your response EXACTLY as:
COMPLIANCE_SCORE: [0-10]

ASSESSMENT:
[Your assessment]

IMPROVEMENTS:
1. [Improvement 1]
2. [Improvement 2]
3. [Improvement 3]

REVISED_SECTION:
[Complete revised content]"""

    def create_section_synthesis_prompt(
        self,
        section_title: str,
        responses: List[Dict[str, Any]]
    ) -> str:
        """Synthesize multiple reviewer responses for a section."""
        responses_text = ""
        for i, resp in enumerate(responses, 1):
            responses_text += f"\n=== REVIEWER {i} (from {resp['model']}) ===\n{resp['response']}\n"

        return f"""You are synthesizing feedback from multiple technical reviewers for the section: "{section_title}"
{responses_text}

Create a unified review that:
1. Averages the compliance scores from all reviewers
2. Identifies consensus improvements (mentioned by 2+ reviewers)
3. Notes any significant disagreements between reviewers
4. Produces a single revised section incorporating the best suggestions

Format your response EXACTLY as:
CONSENSUS_SCORE: [average score 0-10]

SYNTHESIZED FEEDBACK:
[Combined feedback from all reviewers, noting consensus and disagreements]

FINAL SUGGESTED VERSION:
[The synthesized best version of the section]"""

    # =========================================================================
    # RESPONSE PARSING
    # =========================================================================

    def parse_section_review_response(self, response: str) -> Dict[str, Any]:
        """
        Parse a section review response to extract compliance score,
        assessment, improvements, and revised section.
        """
        result = {
            'compliance_score': None,
            'assessment': '',
            'improvements': [],
            'revised_section': ''
        }

        # Extract compliance score
        score_match = re.search(
            r'COMPLIANCE_SCORE:\s*(\d+(?:\.\d+)?)',
            response,
            re.IGNORECASE
        )
        if score_match:
            try:
                score = float(score_match.group(1))
                result['compliance_score'] = min(score / 10.0, 1.0)  # Normalize to 0-1
            except ValueError:
                pass

        # Extract assessment
        assessment_match = re.search(
            r'ASSESSMENT:\s*(.*?)(?=IMPROVEMENTS:|REVISED_SECTION:|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if assessment_match:
            result['assessment'] = assessment_match.group(1).strip()

        # Extract improvements
        improvements_match = re.search(
            r'IMPROVEMENTS:\s*(.*?)(?=REVISED_SECTION:|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if improvements_match:
            improvements_text = improvements_match.group(1).strip()
            result['improvements'] = re.findall(
                r'\d+\.\s*(.+?)(?=\d+\.|$)',
                improvements_text,
                re.DOTALL
            )
            result['improvements'] = [i.strip() for i in result['improvements'] if i.strip()]

        # Extract revised section
        revised_match = re.search(
            r'REVISED_SECTION:\s*(.*?)$',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if revised_match:
            result['revised_section'] = revised_match.group(1).strip()

        # Fallback: if no structure found, use whole response
        if not result['revised_section'] and not result['assessment']:
            result['revised_section'] = response.strip()

        return result

    def parse_synthesis_response(self, response: str) -> Dict[str, Any]:
        """Parse a synthesis response to extract consensus score, feedback, and suggested content."""
        result = {
            'compliance_score': None,
            'feedback': '',
            'suggested_content': ''
        }

        # Extract consensus score
        score_match = re.search(
            r'CONSENSUS_SCORE:\s*(\d+(?:\.\d+)?)',
            response,
            re.IGNORECASE
        )
        if score_match:
            try:
                score = float(score_match.group(1))
                result['compliance_score'] = min(score / 10.0, 1.0)
            except ValueError:
                pass

        # Extract feedback
        feedback_match = re.search(
            r'SYNTHESIZED FEEDBACK:\s*(.*?)(?=FINAL SUGGESTED VERSION:|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if feedback_match:
            result['feedback'] = feedback_match.group(1).strip()

        # Extract suggested content
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
    # DIFF GENERATION
    # =========================================================================

    def generate_unified_diff(
        self,
        original: str,
        revised: str,
        context_lines: int = 3
    ) -> str:
        """Generate a unified diff between original and revised content."""
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

    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================

    async def process_section_async(
        self,
        section_id: str,
        section_title: str,
        section_content: str,
        knowledge_context: str,
        full_outline: str,
        preceding_sections: Optional[List[Dict]] = None,
        callback: Optional[callable] = None
    ) -> SectionReviewResult:
        """
        Main async method for processing a single report section.

        Follows the same 5-stage pattern as ChurnEngine.process_iteration_async:
        1. Create prompt
        2. Gather responses from all models
        3. Synthesize responses
        4. Parse synthesis (extract compliance score + revised content)
        5. Generate diff
        """
        start_time = time.time()
        logger.info(f"=== Starting report section review: '{section_title}' ===")
        logger.debug(f"Content length: {len(section_content)} chars")
        logger.debug(f"Knowledge context length: {len(knowledge_context)} chars")
        logger.debug(f"Models to query: {self.models}")

        # Stage 1: Create the review prompt
        logger.info("Stage 1: Creating section review prompt")
        if callback:
            callback(None, "creating_prompt", section_title)

        prompt = self.create_section_review_prompt(
            section_title=section_title,
            section_content=section_content,
            knowledge_context=knowledge_context,
            full_outline=full_outline,
            preceding_sections=preceding_sections
        )
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

        synthesis_prompt = self.create_section_synthesis_prompt(section_title, raw_responses)
        logger.debug(f"Synthesis prompt length: {len(synthesis_prompt)} chars")

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

        # Stage 4: Parse synthesis and extract results
        logger.info("Stage 4: Parsing synthesis response")
        if callback:
            callback(None, "parsing", None)

        parsed_synthesis = self.parse_synthesis_response(synthesis_response.response)
        synthesized_feedback = parsed_synthesis['feedback']
        suggested_content = parsed_synthesis['suggested_content']
        compliance_score = parsed_synthesis['compliance_score']

        logger.debug(f"Parsed feedback: {len(synthesized_feedback)} chars")
        logger.debug(f"Parsed suggested content: {len(suggested_content)} chars")
        logger.debug(f"Compliance score: {compliance_score}")

        # If no suggested content from synthesis, try individual responses
        if not suggested_content:
            logger.warning("No suggested content from synthesis, extracting from individual responses")
            parsed = self.parse_section_review_response(responses[0].response)
            suggested_content = parsed.get('revised_section', section_content)
            if compliance_score is None:
                compliance_score = parsed.get('compliance_score')
            logger.debug(f"Extracted suggested content: {len(suggested_content)} chars")

        # Default compliance score if still None
        if compliance_score is None:
            compliance_score = 0.5
            logger.warning("No compliance score found, defaulting to 0.5")

        # Stage 5: Generate diff
        logger.info("Stage 5: Generating diff")
        if callback:
            callback(None, "generating_diff", None)

        content_diff = self.generate_unified_diff(section_content, suggested_content)
        logger.debug(f"Generated diff: {len(content_diff)} chars")

        processing_time = time.time() - start_time

        if callback:
            callback(None, "complete", None)

        logger.info(f"=== Section review completed in {processing_time:.1f}s ===")

        return SectionReviewResult(
            section_id=section_id,
            section_title=section_title,
            original_content=section_content,
            revised_content=suggested_content,
            synthesized_feedback=synthesized_feedback,
            content_diff=content_diff,
            compliance_score=compliance_score,
            raw_responses=raw_responses,
            processing_time=processing_time
        )
