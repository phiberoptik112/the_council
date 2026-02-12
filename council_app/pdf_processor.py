"""
PDF-to-Markdown processor using Ollama vision models.

Designed for RAM-constrained environments (Raspberry Pi):
- Processes one page at a time
- Renders at controlled DPI
- Explicit memory cleanup between pages
- Streaming responses to avoid buffering

This is a single-model processor (not a council process).
"""
import base64
import gc
import io
import json
import logging
import time

import aiohttp
import fitz  # PyMuPDF

from django.conf import settings

logger = logging.getLogger('council_app.pdf_processor')

# ============================================================================
# Prompt for vision model
# ============================================================================

MARKDOWN_CONVERSION_PROMPT = """Convert this document page image to Markdown format.

Rules:
- Output ONLY valid Markdown, no commentary or explanation
- Preserve document structure: headings, lists, tables, bold/italic
- Use appropriate heading levels (# ## ### etc.) based on font size
- Convert tables to Markdown table syntax
- Describe images or figures with a brief alt-text description in brackets
- Maintain the reading order of the document
- Skip repetitive headers and footers (page numbers, document titles in margins)
- If text is unclear or partially visible, include your best reading in [brackets]
- For charts or diagrams, describe the key information they convey

Output the Markdown now:"""


class PDFPageProcessor:
    """
    Processes PDF pages one at a time through an Ollama vision model
    to extract Markdown text.
    """

    def __init__(
        self,
        ollama_url: str = None,
        model_name: str = None,
        dpi: int = None,
        num_predict: int = None,
        timeout: int = None,
        keep_alive: str = None,
    ):
        config = getattr(settings, 'COUNCIL_CONFIG', {})
        pdf_config = getattr(settings, 'PDF_PROCESSING', {})

        self.ollama_url = ollama_url or config.get('OLLAMA_URL', 'http://localhost:11434')
        self.model_name = model_name or pdf_config.get('DEFAULT_VISION_MODEL', 'moondream')
        self.dpi = dpi or pdf_config.get('DEFAULT_DPI', 150)
        self.num_predict = num_predict or pdf_config.get('NUM_PREDICT', 4096)
        self.timeout = timeout or pdf_config.get('PAGE_TIMEOUT', 600)
        self.keep_alive = keep_alive or config.get('KEEP_ALIVE', '5m')

        logger.info(
            f"PDFPageProcessor initialized: model={self.model_name}, "
            f"dpi={self.dpi}, timeout={self.timeout}s"
        )

    # ========================================================================
    # PDF operations (synchronous, one page at a time)
    # ========================================================================

    @staticmethod
    def extract_page_count(pdf_path: str) -> int:
        """Open the PDF, return page count, close immediately."""
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count

    def render_page_to_base64(self, pdf_path: str, page_num: int) -> str:
        """
        Render a single PDF page to a base64-encoded PNG string.

        Opens the PDF, renders only the requested page, then closes.
        This keeps memory usage to a single page at a time.

        Args:
            pdf_path: Path to the PDF file
            page_num: 0-based page index

        Returns:
            Base64-encoded PNG image string
        """
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_num]
            # Render at controlled DPI using a zoom matrix
            zoom = self.dpi / 72.0  # 72 DPI is the PDF default
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert pixmap to PNG bytes
            png_bytes = pix.tobytes("png")

            # Base64 encode
            b64_str = base64.b64encode(png_bytes).decode('utf-8')

            # Clean up pixmap explicitly
            del pix
            del png_bytes

            logger.debug(
                f"Rendered page {page_num} at {self.dpi} DPI, "
                f"base64 length: {len(b64_str)} chars"
            )
            return b64_str
        finally:
            doc.close()

    # ========================================================================
    # Ollama vision API (async, streaming)
    # ========================================================================

    async def convert_page_to_markdown(self, base64_image: str) -> str:
        """
        Send a page image to the Ollama vision model and get Markdown back.

        Uses streaming to avoid buffering the entire response in memory.

        Args:
            base64_image: Base64-encoded PNG image

        Returns:
            Extracted Markdown text
        """
        api_url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": MARKDOWN_CONVERSION_PROMPT,
            "images": [base64_image],
            "stream": True,
            "options": {
                "num_predict": self.num_predict,
            },
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive

        logger.debug(f"Sending page to {self.model_name} at {api_url}")

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        markdown_text = ""

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    raise RuntimeError(
                        f"Ollama returned status {resp.status}: {error_body[:500]}"
                    )

                # Stream response: collect NDJSON chunks
                async for line in resp.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            markdown_text += data.get("response", "")
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue

        logger.debug(f"Vision model returned {len(markdown_text)} chars of markdown")
        return markdown_text.strip()

    # ========================================================================
    # High-level orchestration
    # ========================================================================

    async def process_single_page(
        self,
        pdf_path: str,
        page_num: int,
        collection_name: str = None,
        metadata: dict = None,
    ) -> str:
        """
        Full pipeline for a single page: render -> vision model -> store.

        Args:
            pdf_path: Path to the PDF file
            page_num: 0-based page index
            collection_name: Optional ChromaDB collection to store results in
            metadata: Optional metadata dict for ChromaDB storage

        Returns:
            Extracted markdown text
        """
        start_time = time.time()

        # Step 1: Render page to base64 image
        logger.info(f"Processing page {page_num}: rendering to image...")
        b64_image = self.render_page_to_base64(pdf_path, page_num)

        # Step 2: Send to vision model
        logger.info(f"Processing page {page_num}: sending to vision model...")
        markdown = await self.convert_page_to_markdown(b64_image)

        # Step 3: Clean up the base64 image from memory
        del b64_image

        # Step 4: Store in ChromaDB if collection provided
        if collection_name and markdown:
            logger.info(f"Processing page {page_num}: storing in ChromaDB...")
            from . import vector_store
            page_metadata = dict(metadata or {})
            page_metadata['page_number'] = page_num
            vector_store.add_page(
                collection_name=collection_name,
                page_num=page_num,
                text=markdown,
                metadata=page_metadata,
            )

        processing_time = time.time() - start_time
        logger.info(
            f"Page {page_num} complete: {len(markdown)} chars in {processing_time:.1f}s"
        )

        # Step 5: Explicit garbage collection to free RAM
        self.cleanup_page_memory()

        return markdown

    @staticmethod
    def cleanup_page_memory():
        """Explicit garbage collection hint after processing a page."""
        gc.collect()
