
from openai import OpenAI as OpenAIClient
import re
import json
import os
import tiktoken
from transformers import AutoTokenizer
import requests
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.cloud import aiplatform

class APIConfig:
    def __init__(self):
        # Initialize with environment variables or defaults
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "")
        self.qwen3_api_key = os.getenv("QWEN3_API_KEY", "")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_project = os.getenv("GEMINI_PROJECT", "")
        self.gemini_location = os.getenv("GEMINI_LOCATION", "")
        self.local_api_key = os.getenv("LOCAL_API_KEY", "dummy")
        self.local_base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")

api_config = APIConfig()

def init_tokenizers():
    global QWEN_TOKENIZER
    QWEN_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", token=api_config.hf_token)


def safe_json_loads(s: str):
    if not isinstance(s, str):
        raise TypeError(f"Input must be a string, but got {type(s)}")

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[\s\S]+\}', s)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Found JSON-like block but failed to parse: {e}")

    raise ValueError("No valid JSON object found in string.")



class LLMAPIError(Exception):
    """Custom exception for LLM API call failures."""
    pass

def call_llm_api(
    model: str,
    api_key: str,
    base_url: str,
    user_message: str,
    system_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 3072,
    json_mode: bool = True
) -> str:
   
    if not api_key:
        raise ValueError("API key cannot be empty.")
    if not base_url:
        raise ValueError("Base URL cannot be empty.")

    try:
        client = OpenAIClient(api_key=api_key, base_url=base_url)

        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature, # temperture is not used for GPT-5-mini
            "max_completion_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)
        # print(f"resp: {resp}")
        text = resp.choices[0].message.content or ""
        return text

    except Exception as e:
        error_message = f"API call to model '{model}' failed: {e}"
        print(f"[!] {error_message}")
        raise LLMAPIError(error_message) from e


def call_qwen3(user_message,
                    system_prompt) -> str:
    api_key = api_config.qwen3_api_key or api_config.local_api_key
    base_url = api_config.local_base_url

    TOKEN_LIMIT = 40960
    RESERVED_FOR_COMPLETION = 3072
    SAFETY_MARGIN = 200 
    MAX_PROMPT_TOKENS = TOKEN_LIMIT - RESERVED_FOR_COMPLETION - SAFETY_MARGIN

    system_prompt_tokens = QWEN_TOKENIZER.encode(system_prompt)
    user_message_tokens = QWEN_TOKENIZER.encode(user_message)
    total_prompt_tokens_len = len(system_prompt_tokens) + len(user_message_tokens)
    
    final_user_message = user_message

    if total_prompt_tokens_len > MAX_PROMPT_TOKENS:
        user_message_token_budget = MAX_PROMPT_TOKENS - len(system_prompt_tokens)
        
        if user_message_token_budget <= 0:
            raise ValueError("System prompt is too long, no space left for user message.")

        truncated_user_tokens = user_message_tokens[:user_message_token_budget]
        
        final_user_message = QWEN_TOKENIZER.decode(truncated_user_tokens, skip_special_tokens=True)
        
        original_len = len(user_message) 
        truncated_len = len(final_user_message)
        print(f"\n[Qwen] Truncated user message from {original_len} chars to {truncated_len} chars (approx. {len(truncated_user_tokens)} tokens).")
        total_prompt_tokens_len = len(system_prompt_tokens) + len(truncated_user_tokens)

    available_for_completion = TOKEN_LIMIT - total_prompt_tokens_len - SAFETY_MARGIN
    max_tokens = min(RESERVED_FOR_COMPLETION, max(1, available_for_completion))

    result = call_llm_api(
        model="Qwen/Qwen3-32B",
        api_key=api_key,
        base_url=base_url,
        user_message=final_user_message,
        system_prompt=system_prompt,
        json_mode=True,
        max_tokens=max_tokens
    )

    try:
        return safe_json_loads(result), final_user_message
    except:
        return None, final_user_message

def call_deepseek(user_message,
                    system_prompt) -> str:

    api_key = api_config.deepseek_api_key
    base_url = api_config.deepseek_base_url

    TOKEN_LIMIT = 63000
    RESERVED_FOR_COMPLETION = 4096
    SAFETY_MARGIN = 500 
    MAX_PROMPT_TOKENS = TOKEN_LIMIT - RESERVED_FOR_COMPLETION - SAFETY_MARGIN

    encoding = tiktoken.get_encoding("cl100k_base")

    system_prompt_tokens = encoding.encode(system_prompt)
    user_message_tokens = encoding.encode(user_message)
    total_prompt_tokens_len = len(system_prompt_tokens) + len(user_message_tokens)

    final_user_message = user_message

    if total_prompt_tokens_len > MAX_PROMPT_TOKENS:
        print(f"\n[!] Warning: Total prompt tokens ({total_prompt_tokens_len}) exceed the limit ({MAX_PROMPT_TOKENS}).")
        
        user_message_token_budget = MAX_PROMPT_TOKENS - len(system_prompt_tokens)
        
        if user_message_token_budget <= 0:
            raise ValueError("System prompt is too long, no space left for user message.")

        truncated_user_tokens = user_message_tokens[:user_message_token_budget]

        final_user_message = encoding.decode(truncated_user_tokens)
        original_len = len(user_message)
        truncated_len = len(final_user_message)
        print(f"\n[!] Truncated user message from {original_len} chars to {truncated_len} chars (approx. {len(truncated_user_tokens)} tokens).")
        total_prompt_tokens_len = len(system_prompt_tokens) + len(truncated_user_tokens)

    available_for_completion = TOKEN_LIMIT - total_prompt_tokens_len - SAFETY_MARGIN
    max_tokens = min(RESERVED_FOR_COMPLETION, max(1, available_for_completion))

    result = call_llm_api(
            model="deepseek-chat",
            api_key=api_key,
            base_url=base_url,
            user_message=final_user_message, 
            system_prompt=system_prompt,
            temperature=0.0,
            json_mode=False,
            max_tokens=max_tokens
        )
    try:
        return safe_json_loads(result), final_user_message
    except:
        return None, final_user_message
    
    
def call_llama(user_message: str, 
               system_prompt: str) -> str:
    api_key = api_config.local_api_key
    base_url = api_config.local_base_url
    
    TOKEN_LIMIT = 32768
    RESERVED_FOR_COMPLETION = 4096   
    SAFETY_MARGIN = 500             
    MAX_PROMPT_TOKENS = TOKEN_LIMIT - RESERVED_FOR_COMPLETION - SAFETY_MARGIN

    encoding = tiktoken.get_encoding("cl100k_base")
    
    system_prompt_tokens = encoding.encode(system_prompt)
    user_message_tokens = encoding.encode(user_message)
    total_prompt_tokens_len = len(system_prompt_tokens) + len(user_message_tokens)
    
    final_user_message = user_message

    if total_prompt_tokens_len > MAX_PROMPT_TOKENS:
        user_message_token_budget = MAX_PROMPT_TOKENS - len(system_prompt_tokens)
        
        if user_message_token_budget <= 0:
            raise ValueError("System prompt is too long, no space left for user message.")

        truncated_user_tokens = user_message_tokens[:user_message_token_budget]
        
        final_user_message = encoding.decode(truncated_user_tokens)
        
        original_len = len(user_message) 
        truncated_len = len(final_user_message)
        print(f"\n[Llama] Truncated user message from {original_len} chars to {truncated_len} chars (approx. {len(truncated_user_tokens)} tokens).")
        total_prompt_tokens_len = len(system_prompt_tokens) + len(truncated_user_tokens)

    available_for_completion = TOKEN_LIMIT - total_prompt_tokens_len - SAFETY_MARGIN
    max_tokens = min(RESERVED_FOR_COMPLETION, max(1, available_for_completion))

    result = call_llm_api(
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_key=api_key,
        base_url=base_url,
        user_message=final_user_message,
        system_prompt=system_prompt,
        temperature=0.0,
        json_mode=False, 
        max_tokens=max_tokens
    )

    try:
        return safe_json_loads(result), final_user_message
    except:
        return None, final_user_message


def call_coder(user_message: str, 
               system_prompt: str) -> str:
    api_key = api_config.local_api_key
    base_url = api_config.local_base_url
    
    TOKEN_LIMIT = 32768  
    RESERVED_FOR_COMPLETION = 3072
    SAFETY_MARGIN = 500 
    MAX_PROMPT_TOKENS = TOKEN_LIMIT - RESERVED_FOR_COMPLETION - SAFETY_MARGIN

    system_prompt_tokens = QWEN_TOKENIZER.encode(system_prompt)
    user_message_tokens = QWEN_TOKENIZER.encode(user_message)
    total_prompt_tokens_len = len(system_prompt_tokens) + len(user_message_tokens)
    
    final_user_message = user_message

    if total_prompt_tokens_len > MAX_PROMPT_TOKENS:
        user_message_token_budget = MAX_PROMPT_TOKENS - len(system_prompt_tokens)
        
        if user_message_token_budget <= 0:
            raise ValueError("System prompt is too long, no space left for user message.")

        truncated_user_tokens = user_message_tokens[:user_message_token_budget]
        
        final_user_message = QWEN_TOKENIZER.decode(truncated_user_tokens, skip_special_tokens=True)
        
        original_len = len(user_message) 
        truncated_len = len(final_user_message)
        print(f"\n[Qwen] Truncated user message from {original_len} chars to {truncated_len} chars (approx. {len(truncated_user_tokens)} tokens).")
        total_prompt_tokens_len = len(system_prompt_tokens) + len(truncated_user_tokens)

    available_for_completion = TOKEN_LIMIT - total_prompt_tokens_len - SAFETY_MARGIN
    max_tokens = min(RESERVED_FOR_COMPLETION, max(1, available_for_completion))

    result = call_llm_api(
        model="Qwen/Qwen2.5-Coder-14B-Instruct",
        api_key=api_key,
        base_url=base_url,
        user_message=final_user_message,
        system_prompt=system_prompt,
        temperature=0.0,
        json_mode=False,
        max_tokens=max_tokens
    )
    
    try:
        return safe_json_loads(result), final_user_message
    except:
        return None, final_user_message


def call_gpt(user_message: str, system_prompt: str) -> str:

    api_key = api_config.openai_api_key
    base_url = api_config.openai_base_url

    TOKEN_LIMIT = 128000
    RESERVED_FOR_COMPLETION = 4096
    SAFETY_MARGIN = 500
    MAX_PROMPT_TOKENS = TOKEN_LIMIT - RESERVED_FOR_COMPLETION - SAFETY_MARGIN

    encoding = tiktoken.get_encoding("cl100k_base")
    system_prompt_tokens = encoding.encode(system_prompt)
    user_message_tokens = encoding.encode(user_message)
    total_prompt_tokens_len = len(system_prompt_tokens) + len(user_message_tokens)

    final_user_message = user_message
    if total_prompt_tokens_len > MAX_PROMPT_TOKENS:
        user_message_token_budget = MAX_PROMPT_TOKENS - len(system_prompt_tokens)
        if user_message_token_budget <= 0:
            raise ValueError("System prompt too long, no space left for user message.")
        truncated_user_tokens = user_message_tokens[:user_message_token_budget]
        final_user_message = encoding.decode(truncated_user_tokens)

    available_for_completion = TOKEN_LIMIT - total_prompt_tokens_len - SAFETY_MARGIN
    max_tokens = min(RESERVED_FOR_COMPLETION, max(1, available_for_completion))

    result = call_llm_api(
        model="gpt-4o-mini", # change to gpt-5-mini if needed
        api_key=api_key,
        base_url=base_url,
        user_message=final_user_message,
        system_prompt=system_prompt,
        temperature=0.0,
        json_mode=False,
        max_tokens=max_tokens
    )

    try:
        return safe_json_loads(result), final_user_message
    except:
        return result, final_user_message

def call_gemini(user_message: str,
                system_prompt: str):
    genai.configure(api_key=api_config.gemini_api_key)

    aiplatform.init(project=api_config.gemini_project, location=api_config.gemini_location)

    TOKEN_LIMIT = 1048576
    RESERVED_FOR_COMPLETION = 8192
    SAFETY_MARGIN = 500
    MAX_PROMPT_TOKENS = TOKEN_LIMIT - RESERVED_FOR_COMPLETION - SAFETY_MARGIN

    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    system_prompt_tokens = estimate_tokens(system_prompt)
    user_message_tokens = estimate_tokens(user_message)
    total_prompt_tokens_len = system_prompt_tokens + user_message_tokens

    final_user_message = user_message

    if total_prompt_tokens_len > MAX_PROMPT_TOKENS:
        user_message_token_budget = MAX_PROMPT_TOKENS - system_prompt_tokens
        
        if user_message_token_budget <= 0:
            raise ValueError("System prompt is too long, no space left for user message.")
        
        max_user_chars = user_message_token_budget * 4
        final_user_message = user_message[:max_user_chars]
        
        original_len = len(user_message)
        truncated_len = len(final_user_message)
        print(f"\n[Gemini] Truncated user message from {original_len} chars to {truncated_len} chars (approx. {estimate_tokens(final_user_message)} tokens).")
    
    current_prompt_tokens = estimate_tokens(system_prompt + final_user_message)
    available_for_completion = TOKEN_LIMIT - current_prompt_tokens - SAFETY_MARGIN
    max_output_tokens = min(RESERVED_FOR_COMPLETION, max(1, available_for_completion))

    model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt,
        )

    generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            top_p=0.95,
            top_k=40
        )

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    response = model.generate_content(
            final_user_message,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

    if response.candidates and len(response.candidates) > 0:
        result = response.candidates[0].content.parts[0].text
    else:
        raise LLMAPIError("No response generated from Gemini")

    try:
        return safe_json_loads(result), final_user_message
    except:
        return result, final_user_message

def call_claude(user_message: str,
                system_prompt: str,
                cache_control: bool = True):


    api_key = api_config.anthropic_api_key
    base_url = api_config.anthropic_base_url

    TOKEN_LIMIT = 180000
    RESERVED_FOR_COMPLETION = 4096
    SAFETY_MARGIN = 500
    MAX_PROMPT_TOKENS = TOKEN_LIMIT - RESERVED_FOR_COMPLETION - SAFETY_MARGIN

    encoding = tiktoken.get_encoding("cl100k_base")
    system_prompt_tokens = encoding.encode(system_prompt)
    user_message_tokens = encoding.encode(user_message)

    total_prompt_tokens_len = len(system_prompt_tokens) + len(user_message_tokens)

    final_user_message = user_message
    if total_prompt_tokens_len > MAX_PROMPT_TOKENS:
        user_message_token_budget = MAX_PROMPT_TOKENS - len(system_prompt_tokens)
        if user_message_token_budget <= 0:
            raise ValueError("System prompt + context too long, no space left for user message.")
        truncated_user_tokens = user_message_tokens[:user_message_token_budget]
        final_user_message = encoding.decode(truncated_user_tokens)
        print(f"[Claude] Truncated user message from {len(user_message)} chars "
              f"to {len(final_user_message)} chars (approx. {len(truncated_user_tokens)} tokens).")

    available_for_completion = TOKEN_LIMIT - total_prompt_tokens_len - SAFETY_MARGIN
    max_tokens = min(RESERVED_FOR_COMPLETION, max(1, available_for_completion))

    cache_control = {"type": "ephemeral"} if cache_control else None

    payload = {
        "model": "claude-3-7-sonnet-20250219",  # 
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": final_user_message}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    try:
        resp = requests.post(base_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        return safe_json_loads(data["content"][0]["text"]), final_user_message
    except Exception as e:
        print(f"error: {resp.text}")
        return None, final_user_message



        