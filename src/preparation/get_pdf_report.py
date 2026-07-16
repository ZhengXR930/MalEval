#!/usr/bin/env python3
"""
This script crawls threat-intelligence report URLs, saves a reusable "scraped" PDF and
its per-page rendered images, then asks a multimodal LLM to extract:
- Atom evidence as normalized <Action, Asset, Target> triples using a strict controlled vocabulary
- Behavior classifications with rationale and supporting evidence IDs
- A high-level malware behavior summary, with extra focus on any MITRE ATT&CK (Mobile) procedures
"""

import os
import sys
import json
import requests
import base64
import io
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from openai import OpenAI
from playwright.sync_api import sync_playwright
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import get_sample_info_path, make_doc_id, normalize_report_url
from prompts.extraction_prompt import EXTRACTION_PROMPT_TEMPLATE

# Global configuration
OUTPUT_ROOT = Path(__file__).resolve().parent / "reports"
SCRAPE_TIMEOUT_SECONDS = int(os.getenv("SCRAPE_TIMEOUT_SECONDS", "45"))
MAX_PAGES_PER_REQUEST = int(os.getenv("MAX_PAGES_PER_REQUEST", "30"))
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "150"))
# Optional: value for a TCF/consent cookie (same value reused per domain)
CONSENT_COOKIE_VALUE = os.getenv("CONSENT_COOKIE_VALUE")

ALLOWED_ACTIONS = [
    "STEAL",
    "DOWNLOAD",
    "INSTALL",
    "HIDE",
    "OVERLAY",
    "CLICK",
    "ENCRYPT",
    "CONNECT",
    "PREVENT",
    "REQUEST",
    "GRANT",
    "SEND",
    "INJECT",
    "MONITOR",
    "CAPTURE",
    "EXPLOIT"
]
ALLOWED_ASSETS = [
    "CREDENTIALS",
    "FINANCIAL_DATA",
    "SMS",
    "CALL_LOGS",
    "MEDIA",
    "LOCATION",
    "DEVICE_INFO",
    "NOTIFICATIONS",
    "CLIPBOARD",
    "CONTACTS",
    "KEYSTROKES",
    "SENSITIVE_DATA",
    "APP",
    "PAYLOAD",
    "ROOT_PRIVILEGES",
    "ADMIN_PRIVILEGES",
    "CODE",
    "UI_ELEMENT",
    "COMPUTING_RESOURCES",
    "FINANCIAL_DATA",
    None,
]
ALLOWED_TARGETS = [
    "FINANCIAL_APP",
    "SOCIAL_APP",
    "SYSTEM_SETTINGS",
    "C2_SERVER",
    "AD_NETWORK",
    "USER_INTERFACE",
    "SECURITY_SOFTWARE",
    "ACCESSIBILITY_SERVICE",
    "BROWSER",
    "DEVICE_ADMIN",
    "HARDWARE_SENSOR",
    "FILE_SYSTEM",
    "MINING_POOL",
    None,
]

BEHAVIOR_LABELS = [
    "Privacy Stealing", "SMS/CALL", "Remote Control", "Bank Stealing",
    "Ransom", "Abusing Accessibility", "Privilege Escalation",
    "Stealthy Download", "Ads", "Miner", "Tricky Behavior", "Premium Service"
]   

class ReportExtractor:
    """Extracts atom evidence, behavior classifications, and summary from PDF page images via multimodal LLM."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        use_images: bool = True,
        max_pages_per_request: int = MAX_PAGES_PER_REQUEST,
        require_api_key: bool = True,
    ):
        """
        Initialize the ReportExtractor.
        
        Args:
            model: OpenAI model name (defaults to OPENAI_MODEL env var or "gpt-4o")
            use_images: If True, use image-based extraction (recommended, preserves layout and context);
                       if False, use text extraction
            max_pages_per_request: Maximum number of PDF pages to send in a single API request
                                  (for very long documents, pages will be processed in batches)
        """
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5")
        self.use_images = use_images
        self.max_pages_per_request = max_pages_per_request
        api_key = os.getenv("OPENAI_API_KEY")
        if require_api_key and not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for report extraction.")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.output_root = OUTPUT_ROOT
    
    def fetch_html(self, url: str) -> str:
        """
        Fetch HTML content from the given URL.
        
        Args:
            url: URL of the threat intelligence report
            
        Returns:
            HTML content as string
        """
        print(f"Fetching HTML from {url}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    
    def html_to_pdf(self, url: str, output_path: str) -> bool:
        """
        Convert a URL to PDF using Playwright (better rendering than raw HTML string).
        
        Args:
            url: URL to render
            output_path: Path where PDF will be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Rendering URL to PDF with Playwright...")
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)

                # Create context so we can optionally inject consent cookies
                context = browser.new_context()
                
                # Block cookie consent popup requests proactively
                BLOCK_PATTERNS = [
                    "fundingchoicesmessages.google.com",
                    "fundingchoices.google.com",
                    "consent.google.com",
                    "tpc.googlesyndication.com",
                    "pagead2.googlesyndication.com",
                    "doubleclick.net",
                    "googletagmanager.com/gtag/js",
                    "googletagmanager.com/gtm.js",
                    # Common consent management platforms
                    "cookiebot.com",
                    "onetrust.com",
                    "trustarc.com",
                    "quantcast.com",
                    "sourcepoint.com",
                    "cookielaw.org",
                    "cookiepro.com",
                    "iubenda.com",
                    "osano.com",
                    # GDPR/consent related
                    "gdpr-consent",
                    "cookie-consent",
                    "consent-manager",
                    "cookie-banner",
                ]
                
                # Track all requests for debugging
                requested_urls = []
                
                def route_handler(route, request):
                    url_str = request.url
                    requested_urls.append(url_str)
                    if any(pattern in url_str for pattern in BLOCK_PATTERNS):
                        print(f"Blocking request: {url_str[:100]}...")
                        return route.abort()
                    return route.continue_()
                
                context.route("**/*", route_handler)
                
                if CONSENT_COOKIE_VALUE:
                    try:
                        from urllib.parse import urlsplit as _urlsplit

                        s = _urlsplit(url)
                        domain = s.hostname or ""
                        if domain:
                            cookie_domain = "." + domain.lstrip(".")
                            context.add_cookies(
                                [
                                    {
                                        "name": "euconsent-v2",
                                        "value": CONSENT_COOKIE_VALUE,
                                        "domain": cookie_domain,
                                        "path": "/",
                                    }
                                ]
                            )
                            print(f"Injected consent cookie for domain {cookie_domain}")
                    except Exception as e:
                        print(f"Warning: failed to inject consent cookie: {e}")

                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=SCRAPE_TIMEOUT_SECONDS * 1000)
                
                # Debug: Print frames and cookie-related requests
                print("\n=== Page Frames ===")
                print(f"Main: {page.url}, Total: {len(page.frames)}")
                for i, frame in enumerate(page.frames, 1):
                    if frame is not page.main_frame:
                        print(f"  Frame {i}: {frame.url}")
                cookie_urls = [u for u in requested_urls if any(k in u.lower() for k in 
                    ['cookie', 'consent', 'gdpr', 'fundingchoices', 'onetrust', 'cookiebot'])]
                if cookie_urls:
                    print(f"Cookie-related requests: {len(cookie_urls)}")
                    for u in cookie_urls[:3]:  # Show first 3
                        print(f"  {u[:80]}...")
                print("==================\n")
                
                # Remove cookie banners (fallback if route blocking didn't catch them)
                self._handle_consent_overlays(page)
                
                # Wait for content to settle before generating PDF
                page.wait_for_timeout(1200)
                
                page.pdf(path=output_path, format="A4", print_background=True, prefer_css_page_size=True)
                browser.close()
            print(f"PDF saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error converting with Playwright: {e}")
            raise RuntimeError(f"Failed to convert HTML to PDF: {e}")
    
    def _handle_consent_overlays(self, page) -> None:
        """
        Remove cookie consent banners/overlays that might appear in the PDF.
        Only removes actual overlays/banners, never main content.
        """
        try:
            page.wait_for_timeout(2000)  # Wait for banners to load
            
            # Step 1: Hide common cookie banner patterns via CSS
            page.add_style_tag(content="""
                [id*="cookie"], [id*="consent"], [id*="onetrust"], [id*="cookiebot"],
                [class*="cookie-banner"], [class*="cookie-consent"], [class*="gdpr-banner"],
                [class*="consent-banner"], [class*="onetrust"], [class*="cookiebot"],
                [data-cookie], [data-consent], [role="dialog"][aria-modal="true"]
                {
                    display: none !important;
                    visibility: hidden !important;
                    opacity: 0 !important;
                }
            """)
            
            # Step 2: Try clicking accept buttons
            accept_selectors = [
                'button:has-text("Accept"), button:has-text("Agree"), button:has-text("Got it")',
                '#accept-cookies, .accept-cookies, #onetrust-accept-btn-handler',
                'button[id*="accept"], button[aria-label*="Accept"]'
            ]
            for sel in accept_selectors:
                try:
                    if page.locator(sel).first.is_visible(timeout=500):
                        page.locator(sel).first.click(timeout=1000)
                        page.wait_for_timeout(500)
                        break
                except Exception:
                    continue
            
            # Step 3: Remove cookie banners via JavaScript (only overlays, never main content)
            page.evaluate("""
() => {
  // Remove elements by ID/class patterns
  ['cookie-banner', 'cookie-consent', 'gdpr-banner', 'consent-banner',
   'onetrust-consent-sdk', 'onetrust-banner-sdk', 'cookiebot', 'CybotCookiebotDialog']
    .forEach(id => {
      const el = document.getElementById(id);
      if (el) el.remove();
    });
  
  document.querySelectorAll('[class*="cookie-banner"], [class*="cookie-consent"], [class*="onetrust"]')
    .forEach(el => el.remove());
  
  // Remove overlays that mention cookies/consent
  document.querySelectorAll('*').forEach(el => {
    const style = window.getComputedStyle(el);
    const pos = style.position;
    const z = parseInt(style.zIndex || '0', 10);
    const text = (el.textContent || '').toLowerCase();
    const id = (el.id || '').toLowerCase();
    const cls = (el.className || '').toString().toLowerCase();
    
    // Only remove if it's an overlay AND has cookie-related content/attributes
    if ((pos === 'fixed' || pos === 'sticky') && z >= 100) {
      const isCookieRelated = 
        (text.includes('cookie') || text.includes('consent') || text.includes('gdpr')) ||
        (id.includes('cookie') || id.includes('consent')) ||
        (cls.includes('cookie') || cls.includes('consent')) ||
        el.hasAttribute('data-cookie') || el.hasAttribute('data-consent');
      
      // Never remove main content containers
      const tag = el.tagName?.toLowerCase();
      if (isCookieRelated && !['html', 'body', 'main', 'article'].includes(tag)) {
        el.remove();
      }
    }
  });
}
""")
            page.wait_for_timeout(500)
        except Exception as e:
            print(f"Warning: Error in consent overlay handling: {e}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        if not PDF2TEXT_AVAILABLE:
            raise RuntimeError("PyPDF2 not available. Install with: pip install PyPDF2")
        
        text_content = []
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}\n")
        except Exception as e:
            print(f"Warning: Error extracting text from PDF: {e}")
            raise
        
        return "\n".join(text_content)
    
    def pdf_to_images(self, pdf_path: str, images_dir: Path) -> List[Path]:
        """
        Convert PDF pages to PNG images on disk.
        
        Args:
            pdf_path: Path to PDF file
            images_dir: Directory to save images into
            
        Returns:
            List of saved image paths, in page order
        """
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image not available. Install with: pip install pdf2image")
        
        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            images = convert_from_path(pdf_path, dpi=PDF_RENDER_DPI)
            out_paths: List[Path] = []
            for idx, img in enumerate(images, start=1):
                out_path = images_dir / f"page_{idx:04d}.png"
                img.save(out_path, format="PNG")
                out_paths.append(out_path)
            return out_paths
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            raise
    
    def build_extraction_prompt(
        self,
        *,
        malware_name: Optional[str],
        url: str,
        is_image_mode: bool = True,
        page_range: Optional[str] = None,
    ) -> str:
        """
        Build the prompt to instruct the multimodal LLM to extract atom evidence and summary.
        Loads the prompt template from prompts/extraction_prompt.py and fills in dynamic values.
        
        Args:
            malware_name: Optional malware family name
            url: Report URL
            is_image_mode: If True, the prompt will mention that images are provided
            page_range: Optional string describing the page range (e.g., "pages 1-10")
        
        Returns:
            Prompt string for the LLM
        """
        # Prepare dynamic values
        page_note = f" (analyzing {page_range})" if page_range else ""
        
        if is_image_mode:
            image_instruction = (
                "The report is provided as images of PDF pages. The images are in sequential order. "
                "Please read the images directly (multimodal) and use exact quoted text for evidence."
            )
        else:
            image_instruction = "The report content is provided as text below."
        
        malware_name_line = f"Malware family name: {malware_name}" if malware_name else "Malware family name: (unknown)"
        
        # Format the template with dynamic values
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            url=url,
            malware_name=malware_name_line,
            page_note=page_note,
            image_instruction=image_instruction,
            allowed_actions=json.dumps([a for a in ALLOWED_ACTIONS if a is not None], indent=2),
            allowed_assets=json.dumps([a for a in ALLOWED_ASSETS], indent=2),
            allowed_targets=json.dumps([t for t in ALLOWED_TARGETS], indent=2),
            behavior_labels=json.dumps(BEHAVIOR_LABELS, indent=2),
        )
        
        return prompt
    
    def extract_with_llm_text(self, pdf_text: str, *, malware_name: Optional[str], url: str) -> Dict[str, Any]:
        """
        Send PDF text content to LLM API.
        Note: loses formatting/visual context. Image-based extraction is recommended.
        
        Args:
            pdf_text: Extracted text content from PDF
            
        Returns:
            Dictionary containing malware_summary and atom_evidence
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Check API key.")
        
        prompt = self.build_extraction_prompt(malware_name=malware_name, url=url, is_image_mode=False)
        full_prompt = f"{prompt}\n\n--- Report Content ---\n{pdf_text}"
        
        print("Sending report TEXT to LLM for analysis (image-based is recommended)...")
        content = [{"type": "text", "text": full_prompt}]
        return self._call_llm_api(content)
    
    def _encode_image_path(self, image_path: Path) -> str:
        img_bytes = image_path.read_bytes()
        return base64.b64encode(img_bytes).decode("utf-8")

    def extract_with_llm_images(self, image_paths: List[Path], *, malware_name: Optional[str], url: str) -> Dict[str, Any]:
        """
        Send page images to multimodal LLM API for extraction.
        All pages are sent together when possible to preserve document context.
        For very long documents, pages are processed in batches and results are merged.
        
        Args:
            image_paths: List of per-page images in order
            
        Returns:
            Dictionary containing malware_summary and atom_evidence
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Check API key.")
        
        total_pages = len(image_paths)
        if total_pages == 0:
            raise ValueError("No images provided to extract_with_llm_images()")
        
        # If document is too long, process in batches
        if total_pages > self.max_pages_per_request:
            print(f"Document has {total_pages} pages. Processing in batches of {self.max_pages_per_request}...")
            return self._process_long_document(image_paths, malware_name=malware_name, url=url)
        
        # Process all pages together (preserves context and relationships)
        prompt = self.build_extraction_prompt(malware_name=malware_name, url=url, is_image_mode=True)
        
        # Build content array with text prompt and all images in sequential order
        content = [{"type": "text", "text": prompt}]
        for page_num, img_path in enumerate(image_paths, 1):
            img_base64 = self._encode_image_path(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}",
                    "detail": "high"  # Use high detail for better text recognition
                }
            })
        
        print(f"Sending all {total_pages} PDF pages to LLM for analysis (preserving full document context)...")
        return self._call_llm_api(content)
    
    def _process_long_document(self, image_paths: List[Path], *, malware_name: Optional[str], url: str) -> Dict[str, Any]:
        """
        Process a long document in batches and merge results.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dictionary with "summary", "atom_evidence", and "behaviors" (family/type added in process_report)
        """
        all_evidence: List[Dict[str, Any]] = []
        all_behaviors: List[Dict[str, Any]] = []
        merged_summary_parts: List[str] = []
        total_pages = len(image_paths)
        num_batches = (total_pages + self.max_pages_per_request - 1) // self.max_pages_per_request
        
        for batch_num in range(num_batches):
            start_idx = batch_num * self.max_pages_per_request
            end_idx = min(start_idx + self.max_pages_per_request, total_pages)
            batch_images = image_paths[start_idx:end_idx]
            page_range = f"pages {start_idx + 1}-{end_idx}"
            
            print(f"Processing batch {batch_num + 1}/{num_batches}: {page_range}")
            
            prompt = self.build_extraction_prompt(
                malware_name=malware_name,
                url=url,
                is_image_mode=True,
                page_range=page_range,
            )
            content = [{"type": "text", "text": prompt}]
            for img_path in batch_images:
                img_base64 = self._encode_image_path(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": "high"
                    }
                })
            
            batch_result = self._call_llm_api(content)
            
            # Collect summaries from each batch
            batch_summary = batch_result.get("summary", "")
            if batch_summary and isinstance(batch_summary, str):
                merged_summary_parts.append(batch_summary)
            
            # Collect evidence
            for ev in batch_result.get("atom_evidence", []) or []:
                if isinstance(ev, dict):
                    all_evidence.append(ev)
            
            # Collect behaviors
            for behavior in batch_result.get("behaviors", []) or []:
                if isinstance(behavior, dict):
                    all_behaviors.append(behavior)

        # Merge summaries (combine all batch summaries)
        merged_summary = " ".join(merged_summary_parts) if merged_summary_parts else "Summary not available (batch merge)."

        # Deduplicate evidence by raw_text + triple, and reassign IDs
        seen: set[Tuple[str, str, str, str]] = set()
        deduped: List[Dict[str, Any]] = []
        old_id_to_new_id: Dict[str, str] = {}
        ev_counter = 1
        
        for item in all_evidence:
            raw_text = str(item.get("raw_text", "")).strip()
            evidence = item.get("evidence") or {}
            action = str(evidence.get("action", ""))
            asset = str(evidence.get("asset", None))
            target = str(evidence.get("target", None))
            key = (raw_text, action, asset, target)
            if key in seen:
                # Map old ID to existing new ID
                old_id = item.get("id", "")
                if old_id:
                    # Find the existing item with this key
                    for existing_item in deduped:
                        existing_raw = str(existing_item.get("raw_text", "")).strip()
                        existing_ev = existing_item.get("evidence") or {}
                        existing_action = str(existing_ev.get("action", ""))
                        existing_asset = str(existing_ev.get("asset", None))
                        existing_target = str(existing_ev.get("target", None))
                        if (existing_raw, existing_action, existing_asset, existing_target) == key:
                            old_id_to_new_id[old_id] = existing_item.get("id", "")
                            break
                continue
            seen.add(key)
            # Assign new sequential ID
            new_id = f"ev_{ev_counter:02d}"
            old_id = item.get("id", "")
            if old_id:
                old_id_to_new_id[old_id] = new_id
            item_copy = item.copy()
            item_copy["id"] = new_id
            deduped.append(item_copy)
            ev_counter += 1

        # Update behavior supporting_evidence_ids with new IDs and deduplicate behaviors
        seen_behaviors: set[str] = set()
        merged_behaviors: List[Dict[str, Any]] = []
        behavior_label_to_behavior: Dict[str, Dict[str, Any]] = {}
        
        for behavior in all_behaviors:
            label = behavior.get("label", "")
            if not label:
                continue
            
            # Update supporting_evidence_ids
            old_supporting_ids = behavior.get("supporting_evidence_ids", [])
            new_supporting_ids = []
            for old_id in old_supporting_ids:
                new_id = old_id_to_new_id.get(old_id, old_id)
                if new_id not in new_supporting_ids:
                    new_supporting_ids.append(new_id)
            
            behavior_copy = behavior.copy()
            behavior_copy["supporting_evidence_ids"] = new_supporting_ids
            
            # Merge behaviors with same label (combine evidence IDs and rationales)
            if label in behavior_label_to_behavior:
                existing = behavior_label_to_behavior[label]
                # Merge evidence IDs
                existing_ids = set(existing.get("supporting_evidence_ids", []))
                existing_ids.update(new_supporting_ids)
                existing["supporting_evidence_ids"] = sorted(list(existing_ids))
                # Combine rationales if different
                existing_rationale = existing.get("rationale", "")
                new_rationale = behavior_copy.get("rationale", "")
                if new_rationale and new_rationale != existing_rationale:
                    existing["rationale"] = f"{existing_rationale} {new_rationale}".strip()
            else:
                behavior_label_to_behavior[label] = behavior_copy
        
        merged_behaviors = list(behavior_label_to_behavior.values())

        return {
            "summary": merged_summary,
            "atom_evidence": deduped,
            "behaviors": merged_behaviors
        }
    
    def _call_llm_api(self, content: List) -> Dict[str, Any]:
        """
        Call the LLM API with the given content and parse the response.
        
        Args:
            content: List of content items (text and/or images)
            
        Returns:
            Dictionary containing extracted tactics, techniques, and procedures
        """
        response_text = ""  # Initialize to avoid NameError in exception handlers
        try:
            # Newer models (o1, o3, gpt-5) use max_completion_tokens instead of max_tokens
            # and don't support custom temperature values
            is_newer_model = any(x in self.model.lower() for x in ['o1', 'o3', 'gpt-5'])
            api_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            }
            # Only set temperature for models that support it (newer models use default temperature=1)
            if not is_newer_model:
                api_params["temperature"] = 0.0
            
            if is_newer_model:
                api_params["max_completion_tokens"] = 20000
            else:
                api_params["max_tokens"] = 20000
            
            response = self.client.chat.completions.create(**api_params)
            
            if not response.choices:
                raise ValueError("LLM API returned no choices in response")
            
            choice = response.choices[0]
            if not choice.message.content:
                raise ValueError("LLM API returned empty response. No content in choices[0].message.content")
            
            response_text = choice.message.content.strip()
            
            # Check if response_text is empty after stripping
            if not response_text:
                raise ValueError("LLM API returned empty response text after stripping")
            
            # Extract JSON from response, handling multiple formats:
            # 1. Markdown code blocks (```json ... ```)
            # 2. Thinking blocks (<thinking>...</thinking> followed by JSON)
            # 3. Plain JSON
            
            # First, handle thinking blocks (remove <thinking>...</thinking>)
            # Case-insensitive search for thinking tags
            response_lower = response_text.lower()
            if "<thinking>" in response_lower:
                # Find the end of the thinking block (case-insensitive)
                thinking_start_idx = response_lower.find("<thinking>")
                thinking_end_tag = "</thinking>"
                thinking_end_idx = response_lower.find(thinking_end_tag, thinking_start_idx)
                
                if thinking_end_idx != -1:
                    # Extract everything after </thinking>
                    actual_end_idx = thinking_end_idx + len(thinking_end_tag)
                    response_text = response_text[actual_end_idx:].strip()
                    print("Extracted JSON after thinking block")
                else:
                    # If no closing tag, try to find JSON after <thinking>
                    # Look for the first { or [ after <thinking>
                    json_start_after_thinking = -1
                    for i in range(thinking_start_idx + len("<thinking>"), len(response_text)):
                        if response_text[i] in ['{', '[']:
                            json_start_after_thinking = i
                            break
                    if json_start_after_thinking != -1:
                        response_text = response_text[json_start_after_thinking:].strip()
                        print("Extracted JSON after thinking block (no closing tag found)")
            
            # Then handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                # Try to find JSON within code blocks
                parts = response_text.split("```")
                # Look for the part that looks like JSON (starts with {)
                for part in parts:
                    part = part.strip()
                    if part.startswith("{") or part.startswith("["):
                        response_text = part
                        break
                else:
                    # Fallback: take the last code block
                    if len(parts) > 1:
                        response_text = parts[-1].strip()
            
            # Find the JSON object/array in the remaining text
            # Look for the first { or [ that starts a JSON structure
            json_start = -1
            for i, char in enumerate(response_text):
                if char in ['{', '[']:
                    json_start = i
                    break
            
            if json_start != -1:
                response_text = response_text[json_start:]
                # Find the matching closing brace/bracket
                bracket_count = 0
                json_end = -1
                for i, char in enumerate(response_text):
                    if char in ['{', '[']:
                        bracket_count += 1
                    elif char in ['}', ']']:
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_end = i + 1
                            break
                if json_end != -1:
                    response_text = response_text[:json_end]
            
            # Check again after extraction
            if not response_text:
                raise ValueError("Response text is empty after extracting JSON")
            
            # Parse JSON
            extracted_data = json.loads(response_text)
            print("Successfully extracted structured data from report.")
            # LLM returns {"summary": "...", "atom_evidence": [...], "behaviors": [...]}
            # We'll add family_name and type_name in process_report
            return extracted_data
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response text (first 500 chars): {response_text[:500] if response_text else '(empty)'}")
            raise
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            raise
    
    def extract_with_llm(self, pdf_path: str, images_dir: Path, *, malware_name: Optional[str], url: str) -> Dict[str, Any]:
        """
        Extract using images when possible (recommended), otherwise falls back to text.
        
        Args:
            pdf_path: Path to PDF file
            images_dir: Folder to save / reuse rendered images
            
        Returns:
            Dictionary containing malware_summary and atom_evidence
        """
        # Prefer image-based extraction (preserves context and relationships)
        if self.use_images and PDF2IMAGE_AVAILABLE:
            image_paths = sorted(images_dir.glob("page_*.png"))
            if not image_paths:
                image_paths = self.pdf_to_images(pdf_path, images_dir)
            return self.extract_with_llm_images(image_paths, malware_name=malware_name, url=url)
        elif PDF2TEXT_AVAILABLE:
            print("Warning: Using text extraction. Image-based extraction is recommended for better context.")
            pdf_text = self.extract_text_from_pdf(pdf_path)
            return self.extract_with_llm_text(pdf_text, malware_name=malware_name, url=url)
        else:
            raise RuntimeError("Neither PyPDF2 nor pdf2image available. Install pdf2image for best results: pip install pdf2image")
    
    def process_report(
        self,
        *,
        url: str,
        output_dir: Path,
        malware_name: Optional[str],
        analyze: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a single report URL, caching PDF + images in output_dir.
        
        If analyze=True and an API key/client are available, also run LLM analysis and
        write analysis.json. When analyze=False (default), only the scrape/render stage
        is executed (no LLM calls).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / "report.pdf"
        images_dir = output_dir / "images"
        output_json_path = output_dir / "analysis.json"

        if not analyze:
            # Step 1: Create/reuse cached PDF
            if not pdf_path.exists():
                self.html_to_pdf(url, str(pdf_path))
            else:
                print(f"Reusing cached PDF: {pdf_path}")

            # Step 2: Create/reuse rendered images
            image_paths = sorted(images_dir.glob("page_*.png"))
            if image_paths:
                print(f"Reusing cached images: {len(image_paths)} pages")
            else:
                image_paths = self.pdf_to_images(str(pdf_path), images_dir)
                print(f"Rendered {len(image_paths)} pages to images in {images_dir}")

        # Optional Step 3: Extract with LLM
        if analyze:
            if not self.client:
                raise RuntimeError("LLM client not initialized; set OPENAI_API_KEY for analysis.")
            extracted_data = self.extract_with_llm(str(pdf_path), images_dir, malware_name=malware_name, url=url)
            
            # Look up family and type from JSON files
            family_name, type_name = _lookup_family_and_type(url)
            if family_name or type_name:
                print(f"Found in sample info: family={family_name}, type={type_name}")
            
            # Combine LLM summary with looked-up family/type
            llm_summary = extracted_data.get("summary", "")
            final_data = {
                "atom_evidence": extracted_data.get("atom_evidence", []),
                "behaviors": extracted_data.get("behaviors", []),
                "malware_summary": {
                    "type_name": type_name or "",
                    "family_name": family_name or "",
                    "url": url,
                    "summary": llm_summary
                }
            }
            
            self._validate_extracted_data(final_data)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)
            print(f"Saved analysis JSON: {output_json_path}")
            return final_data

        # Render-only mode return payload
        return {
            "url": url,
            "output_dir": str(output_dir),
            "pdf_path": str(pdf_path),
            "num_images": len(image_paths),
        }
    
    def _validate_extracted_data(self, data: Dict) -> None:
        """
        Validate extracted JSON data for required structure and controlled vocabulary usage.
        
        Args:
            data: Extracted data dictionary
        """
        warnings = []
        
        summary = data.get("malware_summary")
        if not isinstance(summary, dict):
            warnings.append("Missing or invalid `malware_summary` object")
        else:
            for k in ("type_name", "family_name", "url", "summary"):
                if k not in summary:
                    warnings.append(f"`malware_summary.{k}` is missing")

        atom = data.get("atom_evidence")
        if not isinstance(atom, list):
            warnings.append("Missing or invalid `atom_evidence` list")
        else:
            evidence_ids = set()
            for idx, item in enumerate(atom, start=1):
                if not isinstance(item, dict):
                    warnings.append(f"atom_evidence[{idx}] is not an object")
                    continue
                # Check for id field
                ev_id = item.get("id")
                if not ev_id or not isinstance(ev_id, str):
                    warnings.append(f"atom_evidence[{idx}].id missing/invalid")
                else:
                    if ev_id in evidence_ids:
                        warnings.append(f"atom_evidence[{idx}].id is duplicate: {ev_id}")
                    evidence_ids.add(ev_id)
                raw_text = item.get("raw_text")
                if not raw_text or not isinstance(raw_text, str):
                    warnings.append(f"atom_evidence[{idx}].raw_text missing/invalid")
                ev = item.get("evidence")
                if not isinstance(ev, dict):
                    warnings.append(f"atom_evidence[{idx}].evidence missing/invalid")
                    continue
                action = ev.get("action")
                asset = ev.get("asset", None)
                target = ev.get("target", None)
                if action not in ALLOWED_ACTIONS:
                    warnings.append(f"atom_evidence[{idx}].evidence.action not in ALLOWED_ACTIONS: {action}")
                if asset not in ALLOWED_ASSETS:
                    warnings.append(f"atom_evidence[{idx}].evidence.asset not in ALLOWED_ASSETS: {asset}")
                if target not in ALLOWED_TARGETS:
                    warnings.append(f"atom_evidence[{idx}].evidence.target not in ALLOWED_TARGETS: {target}")
        
        # Validate behaviors
        behaviors = data.get("behaviors")
        if not isinstance(behaviors, list):
            warnings.append("Missing or invalid `behaviors` list")
        else:
            for idx, behavior in enumerate(behaviors, start=1):
                if not isinstance(behavior, dict):
                    warnings.append(f"behaviors[{idx}] is not an object")
                    continue
                label = behavior.get("label")
                if not label or not isinstance(label, str):
                    warnings.append(f"behaviors[{idx}].label missing/invalid")
                elif label not in BEHAVIOR_LABELS:
                    warnings.append(f"behaviors[{idx}].label not in BEHAVIOR_LABELS: {label}")
                rationale = behavior.get("rationale")
                if not rationale or not isinstance(rationale, str):
                    warnings.append(f"behaviors[{idx}].rationale missing/invalid")
                supporting_ids = behavior.get("supporting_evidence_ids")
                if not isinstance(supporting_ids, list):
                    warnings.append(f"behaviors[{idx}].supporting_evidence_ids missing/invalid (must be array)")
                else:
                    # Check that referenced evidence IDs exist
                    if atom and isinstance(atom, list):
                        valid_ids = {item.get("id") for item in atom if isinstance(item, dict) and item.get("id")}
                        for ref_id in supporting_ids:
                            if ref_id not in valid_ids:
                                warnings.append(f"behaviors[{idx}].supporting_evidence_ids references non-existent ID: {ref_id}")
        
        if warnings:
            print("\n  Validation warnings:")
            for warning in warnings[:10]:  # Show first 10 warnings
                print(f"  - {warning}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more warnings")
            print("\nNote: These are format/vocabulary warnings.")


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _lookup_family_and_type(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Look up family name and type name from JSON files based on URL.
    If URL matches multiple samples, returns the first match.
    
    Args:
        url: URL to look up
        
    Returns:
        Tuple of (family_name, type_name) or (None, None) if not found
    """
    archived_path = Path(get_sample_info_path("malradar"))
    latest_path = Path(get_sample_info_path("new"))
    
    for json_path in [archived_path, latest_path]:
        if not json_path.exists():
            continue
        try:
            data = _load_json(json_path)
            for _, meta in data.items():
                if not isinstance(meta, dict):
                    continue
                sample_url = meta.get("url")
                if sample_url and isinstance(sample_url, str):
                    # Normalize URLs for comparison (remove trailing slashes, etc.)
                    normalized_sample_url = normalize_report_url(sample_url)
                    normalized_input_url = normalize_report_url(url)
                    if normalized_sample_url == normalized_input_url:
                        family = meta.get("family")
                        type_name = meta.get("type")
                        return (str(family) if family else None, str(type_name) if type_name else None)
        except Exception as e:
            print(f"Warning: Error reading {json_path}: {e}")
            continue
    
    return (None, None)


def iter_reports_from_sources(*paths: Path) -> Iterable[Tuple[str, Optional[str]]]:
    """
    Yield (url, malware_name) pairs from the sample info JSON files.
    """
    SKIP_URLS = [
        "https://securelist.com/ztorg-money-for-infecting-your-smartphone/78325/",
        "https://www.trendmicro.com/en_us/research/17/k/toast-overlay-weaponized-install-android-malware-single-attack-chain.html",
        "https://www.welivesecurity.com/2014/06/25/simplocker-new-variants/",
        "https://www.trendmicro.com/en_us/research/19/d/new-version-of-xloader-that-disguises-as-android-apps-and-an-ios-profile-holds-new-links-to-fakespy.html#",
        "https://www.trendmicro.com/en_gb/research/17/g/slocker-mobile-ransomware-starts-mimicking-wannacry.html",
        "https://securelist.com/the-banker-that-encrypted-files/76913/",
        "https://cyble.com/blog/a-new-variant-of-hydra-banking-trojan-targeting-european-banking-users/",
        "https://www.welivesecurity.com/2017/10/13/doublelocker-innovative-android-malware/"
    ]
    SKIP_URLS = set(SKIP_URLS)
    seen: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        data = _load_json(p)
        if not isinstance(data, dict):
            continue
        for _, meta in data.items():
            if not isinstance(meta, dict):
                continue
            url = meta.get("url")
            if not url or not isinstance(url, str):
                continue
            url = url.strip()
            if url in seen:
                continue
            if url in SKIP_URLS:
                continue
            seen.add(url)
            name = meta.get("family") or meta.get("name")
            yield url, (str(name) if name else None)



def main() -> None:
    """
    Usage:
      python get_pdf_report.py
      python get_pdf_report.py <url>
    """
    single_url = sys.argv[1].strip() if len(sys.argv) > 1 else None

    extractor = ReportExtractor(
        use_images=True,
        max_pages_per_request=MAX_PAGES_PER_REQUEST,
        require_api_key=False,
    )

    if single_url:
        doc_id = make_doc_id(single_url)
        out_dir = OUTPUT_ROOT / doc_id
        extractor.process_report(url=single_url, output_dir=out_dir, malware_name=None, analyze=True)
        return

    archived = Path(get_sample_info_path("malradar"))
    latest = Path(get_sample_info_path("new"))

    items = list(iter_reports_from_sources(archived, latest))
    print(f"Found {len(items)} unique report URLs from sources.")
    for i, (url, malware_name) in enumerate(items, start=1):
        doc_id = make_doc_id(url)
        out_dir = OUTPUT_ROOT / doc_id
        print(f"\n[{i}/{len(items)}] Processing {url}")
        try:
            extractor.process_report(url=url, output_dir=out_dir, malware_name=malware_name, analyze=True)
        except Exception as e:
            print(f"[!] Failed: {url}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
