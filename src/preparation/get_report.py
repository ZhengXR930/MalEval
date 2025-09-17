
import requests
import json
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from readability import Document
from urllib.parse import urljoin, urlparse
import re
import os
import hashlib
from PIL import Image
import base64, io
import fitz
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import utils
import unicodedata

path_dict = utils.get_path_dict()


class ReportExtractor:
    def __init__(self, path_dict, folder_name, gt_info_path, max_workers=10):
        self.path_dict = path_dict
        self.folder_name = folder_name
        self.max_workers = max_workers
        self.ua = {"User-Agent":"Mozilla/5.0"}
        self.output_dir = self.path_dict["report"]

        self.gt_info_path = gt_info_path
        with open(self.gt_info_path, "r") as f:
            self.gt_info = json.load(f)


    def _extract_text(self, url):
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
        soup = BeautifulSoup(html, "html.parser")
        
        text = "\n".join(p.get_text(separator=" ", strip=True) for p in soup.find_all("p"))
        
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]', '', text)
        return text

    def _strip_image_wrapping_link(self,md_text):
        pattern = re.compile(
            r'\[!\[(?P<alt>[^\]]*)\]\((?P<src>(?!https?://)[^)]+)\)\]\((?P<href>https?://|/)[^)]+\)'
        )
        return pattern.sub(r'![\g<alt>](\g<src>)', md_text)


    def _sanitize_filename(self,url):
        p = urlparse(url)
        path = p.path
        name = os.path.basename(path) or "image"
        if "." not in name:
            name += ".jpg"
        return name.split("?")[0]

    def _hash_name(self, content, orig_name):
        stem, ext = os.path.splitext(orig_name)
        h = hashlib.sha256(content).hexdigest()[:16]
        return f"{stem}-{h}{ext}"

    def check_site(self, url):
        headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
            }
        try:
            resp = requests.head(url, allow_redirects=True, timeout=10, headers=headers)
            status = resp.status_code
            ctype = resp.headers.get("Content-Type", "").lower()
            if status >= 400:
                return False, f"HTTP {status}"
            if "text/html" not in ctype:
                return False, f"Not HTML: {ctype}"
            return True, f"HTML OK ({status})"
        except requests.exceptions.RequestException as e:
            return False, str(e)

    def _fetch_html(self, url, timeout=30):
        r = requests.get(url, headers=self.ua, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r.text, r.url 

        
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\ufffd]', '', text)
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != "C")
        return text

    def html_to_markdown_with_local_images(self, html, base_url, out_dir="output"):
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        doc = Document(html)
        main_html = doc.summary()
        soup = BeautifulSoup(main_html, "html.parser")

        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if not src:
                img.decompose(); continue
            full = urljoin(base_url, src)

            try:
                ir = requests.get(full, headers=self.ua, timeout=30, stream=True)
                ir.raise_for_status()
                content = ir.content

                orig_name = self._sanitize_filename(full)
                local_name = self._hash_name(content, orig_name)
                local_path = os.path.join(out_dir, 'images', local_name)
                with open(local_path, "wb") as f:
                    f.write(content)

                alt = (img.get("alt") or "").strip()
                if not alt:
                    cap = img.find_next("figcaption")
                    if cap:
                        alt = cap.get_text(strip=True)

                img["src"] = f"images/{local_name}"
                if alt:
                    img["alt"] = alt
            except Exception:
                img.decompose()

        markdown = md(str(soup), heading_style="ATX")
        markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
        markdown = self._strip_image_wrapping_link(markdown)
        markdown = self._clean_text(markdown)
        md_path = os.path.join(out_dir, "page.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        return md_path

    def _read_markdown(self,md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()


    def _extract_images(self,md_text, base_dir):
        IMG_PATTERN = re.compile(r'!\[(.*?)\]\((.*?)\)')  

        images = []
        for alt, path in IMG_PATTERN.findall(md_text):
            if path.startswith("http://") or path.startswith("https://"):
                continue
            abs_path = os.path.join(base_dir, path)
            if os.path.exists(abs_path):
                images.append(abs_path)
        return images

    def _downscale_and_b64(self,img_path, max_side=1400, quality=85):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"


    def _pdf_to_text_and_images(self,url, out_dir="pdf_out", dpi_scale=2.0):
        os.makedirs(out_dir, exist_ok=True)
        pdf_path = os.path.join(out_dir, utils.make_doc_id(url), "doc.pdf")

        r = requests.get(url, headers=self.ua, timeout=60)
        r.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(r.content)

        doc = fitz.open(pdf_path)
        text_parts, img_paths = [], []
        for i, page in enumerate(doc):
            text_parts.append(page.get_text("text"))

            m = fitz.Matrix(dpi_scale, dpi_scale)  
            pix = page.get_pixmap(matrix=m, alpha=False)
            img_p = os.path.join(out_dir, utils.make_doc_id(url), f"page-{i+1}.jpg")
            pix.save(img_p)
            img_paths.append(img_p)

        full_text = "\n\n".join(text_parts).strip()
        markdown = md(full_text, heading_style="ATX")
        markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
        markdown = self._strip_image_wrapping_link(markdown)
        md_path = os.path.join(out_dir, utils.make_doc_id(url), "page.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        return markdown

    def _build_multi_modal_message(self,md_text: str, images: List[str]):
        BEHAVIORS = [
            "privacy stealing", "SMS/CALL", "Remote Control", "Bank Stealing",
            "Ransom", "Abusing Accessibility", "Privilege Escalation",
            "Stealthy Escalation", "Ads", "Miner", "Tricky Behavior", "Premium Service",
        ]
        
        SYSTEM_PROMPT = (
            "You are a mobile malware analyst. Read the provided report (full text + figures) "
            "and assess ONLY the given behavior set. Use EXCLUSIVELY those labels; do not invent new ones. "
            "Mark a behavior as present only with explicit evidence; otherwise mark it as uncertain. "
            "Cite evidence inline using short quotes (≤50 words) from the text and/or [fig:filename] for images. "
            "Keep bullets concise, factual, and source-grounded."
        )

        USER_PROMPT_TEMPLATE = """\
        You are given a malware analysis report and figures. Analyze them based on the provided behavior set.
        Your response MUST be a single, valid JSON object. Do not include any text outside of the JSON.

        Behavior set (CLOSED WORLD -- use EXACT strings, case-sensitive):
        {behavior_list}

        **JSON Output Schema (Example with multiple behaviors):**
        YOU MUST USE THE FOLLOWING JSON OUTPUT SCHEMA:
        {{
        "present_behaviors": [
            {{
            "behavior": "Abusing Accessibility",
            "is_uncertain": false,
            "reason": "The malware presents a deceptive overlay to trick the user into granting Accessibility Service permissions.",
            "citations": ["...presents a deceptive overlay...", "[fig:fig1_permissions.jpg]"]
            }},
            {{
            "behavior": "privacy stealing",
            "is_uncertain": false,
            "reason": "The application was observed sending the user's contact list to a remote C2 server.",
            "citations": ["...sending sensitive information, including contact lists..."]
            }}
        ],
        "overall_summary": "A 3-5 sentence summary of how the malware operates (vector, capabilities, goals), with citations.",
        "notes": "A string containing observations of behaviors NOT in the provided set, or null if there are none."
        }}

        **Rules for JSON content:**
        - For `present_behaviors`:
        - **Create a separate JSON object inside the array for each distinct behavior identified.** Do not combine multiple behaviors into one object.
        - If no behaviors from the set are found, the array MUST be empty (`[]`).
        - `behavior`: Must be an EXACT string from the behavior set.
        - `is_uncertain`: Set to `true` if evidence is ambiguous, otherwise `false`.
        - `citations`: A JSON array of strings. Each string must be a short quote or a figure reference like `[fig:filename]`.
        - For `overall_summary`: Provide a concise summary with citations.
        - For `notes`: If you observe other malicious activities, describe them here. If not, the value should be `null`.

        Analyze the figures and the full text now."""

        # Build user content as separate, manageable pieces
        user_content = []
        
        # Add the main prompt
        behavior_list = "\n".join(f"- {b}" for b in BEHAVIORS)
        user_prompt = USER_PROMPT_TEMPLATE.format(behavior_list=behavior_list)
        user_content.append({"type": "text", "text": user_prompt})
        
        # Add the report text separately
        user_content.append({"type": "text", "text": f"\n\n--- FULL TEXT REPORT ---\n\n{md_text}"})
        
        # Add figure references
        if images:
            figure_text = "\n\n--- FIGURES ---\nLocal figures with identifiers:"
            for p in images:
                if os.path.exists(p):
                    fname = os.path.basename(p)
                    figure_text += f"\n[fig:{fname}]"
            user_content.append({"type": "text", "text": figure_text})
            
            # Add actual images
            for p in images:
                if os.path.exists(p):
                    data_url = self._downscale_and_b64(p)
                    if data_url:
                        user_content.append({"type": "image_url", "image_url": {"url": data_url}})

        final_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        return final_messages

    def _build_message(self,md_text: str):
        """Analyze malware report text and extract behaviors"""
        
        BEHAVIORS = [
            "privacy stealing", "SMS/CALL", "Remote Control", "Bank Stealing",
            "Ransom", "Abusing Accessibility", "Privilege Escalation", 
            "Stealthy Escalation", "Ads", "Miner", "Tricky Behavior", "Premium Service"
        ]
        
        system_prompt = """You are a mobile malware analyst. Analyze the provided malware report and identify behaviors from the given list. Respond only with valid JSON."""
        
        user_prompt = f"""
        Analyze this malware report and identify behaviors from this exact list, also you need to generate a comprehensive summary after the analysis:
        {', '.join(BEHAVIORS)}

        Analysis Focus:
        1. Look for behavior descriptions, avoid over-interpretation
        2. Distinguish between what malware "can do" vs "actually does"
        3. Prioritize behaviors with direct evidence support
        4. Note that behaviors may overlap



        Report text:
        {md_text}

        Return JSON with this structure:
        {{
        "present_behaviors": [
            {{
            "behavior": "exact behavior name MUST be from list",
            "confidence": MUST be one of "low", "medium", or "high".
            "evidence": "A concise string (1-2 sentences) explaining *why* you chose that behavior, citing specific actions from the text."
            }}
        ],
        "summary": "a comprehensive summary describing malware's main functionality, and how it executes",
        }}

        Rules:
        - Only use behaviors from the provided list (exact strings)
        - Include evidence quotes from the report text
        - If no behaviors found, return empty array for present_behaviors
        - Maintain conservative approach - don't include uncertain behaviors
        """

        return system_prompt, user_prompt

    
    def process_single_pdf(self,pdf_url, output_dir):
        markdown = self._pdf_to_text_and_images(pdf_url, output_dir)
        system_prompt, user_prompt = self._build_message(markdown)
        result, _ = utils.get_llm_client(model="gpt")(user_prompt, system_prompt) # change to gpt-5-mini
        result['url'] = pdf_url
        result['status'] = 'success'
        result_path = os.path.join(output_dir, utils.make_doc_id(pdf_url), "analysis_result.json")
        with open(result_path, "w", encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result
    


    def process_single_url(self,url, output_dir):
        """Process a single URL with proper error handling"""
        result_path = os.path.join(output_dir, utils.make_doc_id(url), "analysis_result.json")
 
        try:
            status, msg = self.check_site(url)
            if not status:
                return {"url": url, "status": "failed", "error": f"Site check failed: {msg}"}
            
            html, base_url = self._fetch_html(url)
            doc_id = utils.make_doc_id(url)
            
            # Ensure output directory exists
            doc_output_dir = os.path.join(output_dir, doc_id)
            os.makedirs(doc_output_dir, exist_ok=True)
            
            md_path = self.html_to_markdown_with_local_images(html, base_url, doc_output_dir)
            md_text = self._read_markdown(md_path)
            # print(f"md_text: {md_text}")
            system_prompt, user_prompt = self._build_message(md_text)
            result, final_user_prompt = utils.get_llm_client(model="gpt")(user_prompt, system_prompt)
            # print(f"result: {result}")
            result['url'] = url
            result['status'] = 'success'
            
            # Safe file writing
            
            with open(result_path, "w", encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"result_path: {result_path}")
            return result
            
        except Exception as e:
            return {"url": url, "status": "failed", "error": str(e)}

    def process_all_url(self,apk_list):
        url_list = [self.gt_info[apk_name]["url"] for apk_name in apk_list]

        to_remove = []
        for url in url_list:
            if os.path.exists(os.path.join(self.output_dir, utils.make_doc_id(url), "analysis_result.json")):
                to_remove.append(url)
        url_list = [url for url in url_list if url not in to_remove]
   
        successful_results = []
        failed_results = []
        url_list = list(set(url_list))
        print(f"number of url_list: {len(url_list)} need to be processed")
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_single_url, url, self.output_dir): url 
                      for url in url_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing URLs"):
                url = futures[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per task
                    if result.get('status') == 'success':
                        successful_results.append(result)
                        print(f"Processed {url}")
                    else:
                        failed_results.append(result)
                        print(f"Failed {url}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    failed_results.append({"url": url, "status": "failed", "error": str(e)})
                    print(f"Exception for {url}: {e}")
    
        end_time = time.time()
        print(f"time cost: {end_time - start_time} seconds")
        print(f"number of successful_results: {len(successful_results)}")
        print(f"number of failed_results: {len(failed_results)}")
        with open(os.path.join(path_dict["info"], "successful_results.json"), "w") as f:
            json.dump(successful_results, f, indent=2)
        with open(os.path.join(path_dict["info"], "failed_results.json"), "w") as f:
            json.dump(failed_results, f, indent=2)


if __name__ == "__main__":

    folder_name = "archived"
    if folder_name == "latest":
        json_name = os.path.join(path_dict["info"], "latest_sample_info.json")
    elif folder_name == "archived":
        json_name = os.path.join(path_dict["info"], "archived_sample_info.json")
    else:
        raise ValueError(f"Invalid folder name: {folder_name}")

    apk_list = utils.check_file_exist(folder_name=folder_name)
    gt_info_path = os.path.join(path_dict["info"], json_name)
    report_extractor = ReportExtractor(path_dict, folder_name, gt_info_path)
    report_extractor.process_all_url(apk_list)