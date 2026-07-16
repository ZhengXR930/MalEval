
from openai import OpenAI as OpenAIClient
import re
import json
import os
from pathlib import Path
import tiktoken
import requests
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError:
    AutoTokenizer = None

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    from google.cloud import aiplatform
except ModuleNotFoundError:
    genai = None
    HarmCategory = None
    HarmBlockThreshold = None
    aiplatform = None

def _load_model_registry() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    registry_path = Path(os.getenv("MALEVAL_MODEL_REGISTRY", repo_root / "model_registry.yaml"))
    if not registry_path.exists():
        return {}
    if yaml is None:
        raise ModuleNotFoundError(
            "PyYAML is required to read model_registry.yaml. Install pyyaml or use environment variables only."
        )
    with registry_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("models", {})


def _registry_value(registry: dict, model_name: str, field: str, env_field: str | None = None, default=None):
    config = registry.get(model_name, {})
    env_name = config.get(env_field or f"{field}_env")
    if env_name and os.getenv(env_name):
        return os.getenv(env_name)
    if field in config and config[field] not in (None, ""):
        return config[field]
    return default


class APIConfig:
    def __init__(self):
        self.registry = _load_model_registry()
        self.hf_token = (
            _registry_value(self.registry, "qwen", "tokenizer_hf_token", "tokenizer_hf_token_env")
            or _registry_value(self.registry, "coder", "tokenizer_hf_token", "tokenizer_hf_token_env")
            or os.getenv("HF_TOKEN")
        )
        self.openai_api_key = _registry_value(self.registry, "gpt", "api_key", default=os.getenv("OPENAI_API_KEY"))
        self.openai_base_url = _registry_value(
            self.registry,
            "gpt",
            "base_url",
            default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        self.openai_model = _registry_value(self.registry, "gpt", "model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self.gpt5_model = _registry_value(self.registry, "gpt-5", "model", default="gpt-5")
        self.qwen3_api_key = _registry_value(self.registry, "qwen", "api_key", default=os.getenv("QWEN3_API_KEY"))
        self.qwen3_base_url = _registry_value(self.registry, "qwen", "base_url", default=self.openai_base_url)
        self.qwen3_model = _registry_value(self.registry, "qwen", "model", default="Qwen/Qwen3-32B")
        self.deepseek_api_key = _registry_value(self.registry, "deepseek", "api_key", default=os.getenv("DEEPSEEK_API_KEY"))
        self.deepseek_base_url = _registry_value(self.registry, "deepseek", "base_url", default=os.getenv("DEEPSEEK_BASE_URL"))
        self.deepseek_model = _registry_value(self.registry, "deepseek", "model", default="deepseek-reasoner")
        self.anthropic_api_key = _registry_value(self.registry, "claude", "api_key", default=os.getenv("ANTHROPIC_API_KEY"))
        self.anthropic_base_url = _registry_value(self.registry, "claude", "base_url", default=os.getenv("ANTHROPIC_BASE_URL"))
        self.claude_model = _registry_value(self.registry, "claude", "model", default="claude-3-7-sonnet-20250219")
        self.gemini_api_key = _registry_value(self.registry, "gemini", "api_key", default=os.getenv("GEMINI_API_KEY"))
        self.gemini_project = _registry_value(self.registry, "gemini", "project", default=os.getenv("GEMINI_PROJECT"))
        self.gemini_location = _registry_value(self.registry, "gemini", "location", default=os.getenv("GEMINI_LOCATION"))
        self.gemini_model = _registry_value(self.registry, "gemini", "model", default="gemini-2.5-pro")
        self.local_api_key = os.getenv("LOCAL_API_KEY")
        self.local_base_url = os.getenv("LOCAL_BASE_URL")
        self.huggingface_base_url = _registry_value(self.registry, "llama", "base_url", default=os.getenv("HUGGINGFACE_BASE_URL"))
        self.huggingface_api_key = _registry_value(self.registry, "llama", "api_key", default=os.getenv("HUGGINGFACE_API_KEY"))
        self.llama_model = _registry_value(self.registry, "llama", "model", default="meta-llama/Llama-3.1-8B-Instruct")
        self.coder_api_key = _registry_value(self.registry, "coder", "api_key", default=self.huggingface_api_key)
        self.coder_base_url = _registry_value(self.registry, "coder", "base_url", default=self.huggingface_base_url)
        self.coder_model = _registry_value(self.registry, "coder", "model", default="Qwen/Qwen2.5-Coder-14B-Instruct")

api_config = APIConfig()

QWEN_TOKENIZER = None
USAGE_HOOK = None
DEFAULT_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))


def set_usage_hook(hook):
    global USAGE_HOOK
    USAGE_HOOK = hook


def _usage_to_dict(usage):
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    data = {}
    for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, attr, None)
        if value is not None:
            data[attr] = value
    return data

def get_qwen_tokenizer():
    global QWEN_TOKENIZER
    if QWEN_TOKENIZER is None:
        if AutoTokenizer is None:
            raise ModuleNotFoundError("transformers is required for qwen/coder models. Please install transformers.")
        QWEN_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", token=api_config.hf_token)
    return QWEN_TOKENIZER

def init_tokenizers():
    get_qwen_tokenizer()


def safe_json_loads(s: Any):
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
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "timeout": DEFAULT_REQUEST_TIMEOUT,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""

        if callable(USAGE_HOOK):
            try:
                USAGE_HOOK(
                    {
                        "provider_model": getattr(resp, "model", model),
                        "requested_model": model,
                        "usage": _usage_to_dict(getattr(resp, "usage", None)),
                        "prompt_chars": len(system_prompt) + len(user_message),
                        "response_chars": len(content),
                    }
                )
            except Exception:
                pass

        return content
    except Exception as e:
        raise LLMAPIError(f"Failed to call model '{model}': {e}") from e


def call_qwen3(user_message, system_prompt) -> str:
    api_key = api_config.qwen3_api_key
    base_url = api_config.qwen3_base_url

    TOKEN_LIMIT = 40960
    RESERVED_FOR_COMPLETION = 4096
    SAFETY_MARGIN = 200 
    MAX_PROMPT_TOKENS = TOKEN_LIMIT - RESERVED_FOR_COMPLETION - SAFETY_MARGIN

    tokenizer = get_qwen_tokenizer()
    system_prompt_tokens = tokenizer.encode(system_prompt)
    user_message_tokens = tokenizer.encode(user_message)
    total_prompt_tokens_len = len(system_prompt_tokens) + len(user_message_tokens)
    
    final_user_message = user_message

    if total_prompt_tokens_len > MAX_PROMPT_TOKENS:
        user_message_token_budget = MAX_PROMPT_TOKENS - len(system_prompt_tokens)
        
        if user_message_token_budget <= 0:
            raise ValueError("System prompt is too long, no space left for user message.")

        truncated_user_tokens = user_message_tokens[:user_message_token_budget]
        
        final_user_message = tokenizer.decode(truncated_user_tokens, skip_special_tokens=True)
        total_prompt_tokens_len = len(system_prompt_tokens) + len(truncated_user_tokens)

    available_for_completion = TOKEN_LIMIT - total_prompt_tokens_len - SAFETY_MARGIN
    max_tokens = min(RESERVED_FOR_COMPLETION, max(1, available_for_completion))

    result = call_llm_api(
        model=api_config.qwen3_model,
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

def call_deepseek(user_message: str, system_prompt: str) -> str:

    api_key = api_config.deepseek_api_key
    base_url = api_config.deepseek_base_url

    TOKEN_LIMIT = 64000
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
        total_prompt_tokens_len = len(system_prompt_tokens) + len(truncated_user_tokens)

    available_for_completion = TOKEN_LIMIT - total_prompt_tokens_len - SAFETY_MARGIN
    max_tokens = min(RESERVED_FOR_COMPLETION, max(1, available_for_completion))

    result = call_llm_api(
            model=api_config.deepseek_model,
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
    
    
def call_llama(user_message: str, system_prompt: str) -> str:
    
    api_key = api_config.huggingface_api_key
    base_url = api_config.huggingface_base_url
    
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
        model=api_config.llama_model,
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


def call_coder(user_message: str, system_prompt: str) -> str:
    api_key = api_config.coder_api_key
    base_url = api_config.coder_base_url
    
    TOKEN_LIMIT = 32768  
    RESERVED_FOR_COMPLETION = 4096
    SAFETY_MARGIN = 500 
    MAX_PROMPT_TOKENS = TOKEN_LIMIT - RESERVED_FOR_COMPLETION - SAFETY_MARGIN

    tokenizer = get_qwen_tokenizer()
    system_prompt_tokens = tokenizer.encode(system_prompt)
    user_message_tokens = tokenizer.encode(user_message)
    total_prompt_tokens_len = len(system_prompt_tokens) + len(user_message_tokens)
    
    final_user_message = user_message

    if total_prompt_tokens_len > MAX_PROMPT_TOKENS:
        user_message_token_budget = MAX_PROMPT_TOKENS - len(system_prompt_tokens)
        
        if user_message_token_budget <= 0:
            raise ValueError("System prompt is too long, no space left for user message.")

        truncated_user_tokens = user_message_tokens[:user_message_token_budget]
        
        final_user_message = tokenizer.decode(truncated_user_tokens, skip_special_tokens=True)
        
        original_len = len(user_message) 
        truncated_len = len(final_user_message)
        print(f"\n[Coder] Truncated user message from {original_len} chars to {truncated_len} chars (approx. {len(truncated_user_tokens)} tokens).")
        total_prompt_tokens_len = len(system_prompt_tokens) + len(truncated_user_tokens)

    available_for_completion = TOKEN_LIMIT - total_prompt_tokens_len - SAFETY_MARGIN
    max_tokens = min(RESERVED_FOR_COMPLETION, max(1, available_for_completion))

    result = call_llm_api(
        model=api_config.coder_model,
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


def _call_openai_model(user_message: str, system_prompt: str, model_name: str) -> str:
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
        model=model_name,
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


def call_gpt(user_message: str, system_prompt: str) -> str:
    return _call_openai_model(user_message, system_prompt, api_config.openai_model)


def call_gpt5(user_message: str, system_prompt: str) -> str:
    return _call_openai_model(user_message, system_prompt, api_config.gpt5_model)


def call_gemini(user_message: str, system_prompt: str) -> str:
    if genai is None or aiplatform is None or HarmCategory is None or HarmBlockThreshold is None:
        raise ModuleNotFoundError(
            "google-generativeai and google-cloud-aiplatform are required for gemini model."
        )
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
            model_name=api_config.gemini_model,
            system_instruction=system_prompt,
        )

    generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            top_p=0.95,
            top_k=40
        )

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    response = model.generate_content(
            final_user_message,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and candidate.content and \
           hasattr(candidate.content, 'parts') and candidate.content.parts and len(candidate.content.parts) > 0:
            result = candidate.content.parts[0].text
        else:
            finish_reason = getattr(candidate, 'finish_reason', None)
            safety_ratings = getattr(candidate, 'safety_ratings', None)
            print(f"[Gemini] Response blocked or empty. Finish reason: {finish_reason}, Safety ratings: {safety_ratings}")
            return None, final_user_message
    else:
        print(f"[Gemini] No candidates in response. Prompt feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
        return None, final_user_message

    try:
        return safe_json_loads(result), final_user_message
    except:
        return result, final_user_message


def call_claude(user_message: str,
                system_prompt: str,
                cache_control: bool = True) -> str:


    api_key = api_config.anthropic_api_key
    base_url = api_config.anthropic_base_url

    TOKEN_LIMIT = 200000
    RESERVED_FOR_COMPLETION = 6000
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
        total_prompt_tokens_len = len(system_prompt_tokens) + len(truncated_user_tokens)

    available_for_completion = TOKEN_LIMIT - total_prompt_tokens_len - SAFETY_MARGIN
    max_tokens = min(RESERVED_FOR_COMPLETION, max(1, available_for_completion))

    cache_control = {"type": "ephemeral"} if cache_control else None

    payload = {
        "model": api_config.claude_model,
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

    resp = None
    try:
        resp = requests.post(base_url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        return safe_json_loads(data["content"][0]["text"]), final_user_message
    except Exception as e:
        if resp is not None:
            print(f"error: {resp.text}")
        else:
            print(f"error: {e}")
        return None, final_user_message
