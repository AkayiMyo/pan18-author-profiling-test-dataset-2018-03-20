#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import httpx
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import deque
import threading

def load_rules(rules_path: Path) -> str:
    if not rules_path.exists():
        print(f"Error: rules.md not found at {rules_path}", file=sys.stderr)
        sys.exit(1)
    return rules_path.read_text(encoding="utf-8")

def extract_tweets(xml_path: Path) -> list[str]:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        tweets = []
        for doc in root.findall(".//document"):
            if doc.text:
                tweets.append(doc.text.strip())
        return tweets
    except Exception as e:
        print(f"Error parsing XML file {xml_path}: {e}", file=sys.stderr)
        return []

class TokenRateLimiter:
    def __init__(self, max_tokens_per_minute: int = 10000):
        self.max_tokens = max_tokens_per_minute
        self.history = deque()
        self.lock = threading.Lock()

    def acquire(self, estimated_tokens: int):
        with self.lock:
            now = time.time()
            # Remove entries older than 60 seconds
            while self.history and now - self.history[0][0] > 60.0:
                self.history.popleft()

            current_tokens = sum(t[1] for t in self.history)
            
            while current_tokens + estimated_tokens > self.max_tokens:
                oldest_time, oldest_tokens = self.history[0]
                sleep_time = (oldest_time + 60.0) - time.time()
                if sleep_time > 0:
                    sys.stderr.write(f"\n[Rate Limiter] Approaching limit ({current_tokens + estimated_tokens}/{self.max_tokens} tokens). Sleeping {sleep_time:.2f}s...\n")
                    sys.stderr.flush()
                    time.sleep(sleep_time)
                
                now = time.time()
                while self.history and now - self.history[0][0] > 60.0:
                    self.history.popleft()
                current_tokens = sum(t[1] for t in self.history)

            self.history.append((time.time(), estimated_tokens))

def call_groq(api_key: str, model: str, system_instruction: str, prompt: str, rate_limiter: TokenRateLimiter, response_format: dict = None) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 4096
    }
    if response_format:
        payload["response_format"] = response_format

    
    # Estimate tokens: prompt chars / 4 + system instruction chars / 4 + expected output (approx 1500 tokens)
    estimated_tokens = len(prompt) // 4 + len(system_instruction) // 4 + 1500
    
    # Wait for rate limiter slots
    rate_limiter.acquire(estimated_tokens)

    max_retries = 8
    for attempt in range(max_retries):
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=120.0)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if (e.response.status_code == 429 or 500 <= e.response.status_code < 600) and attempt < max_retries - 1:
                wait_time = 30.0 * (attempt + 1) if e.response.status_code == 429 else 10.0 * (attempt + 1)
                sys.stderr.write(f"\n[Groq HTTP Error {e.response.status_code}] Waiting {wait_time}s before retry...\n")
                sys.stderr.flush()
                time.sleep(wait_time)
                continue
            raise e
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 10.0 * (attempt + 1)
                sys.stderr.write(f"\n[Groq Error: {e}] Waiting {wait_time}s before retry...\n")
                sys.stderr.flush()
                time.sleep(wait_time)
                continue
            raise e
    raise RuntimeError("Failed to call Groq API after retries.")

def main() -> int:
    parser = argparse.ArgumentParser(description="Translate PAN18 English XML files to Myanmar using Groq API.")
    parser.add_argument("--input-dir", default="en/text", help="Directory containing English XML files.")
    parser.add_argument("--output-dir", default="my/text", help="Directory to save Myanmar translated files.")
    parser.add_argument("--model", default="meta-llama/llama-4-scout-17b-16e-instruct", help="Groq model name.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of files to process.")
    parser.add_argument("--ext", default=None, help="Extension of the output files (e.g., .txt or .xml). If None, defaults to match format.")
    parser.add_argument("--format", choices=["xml", "report"], default="xml", help="Output format: 'xml' for translated tweets XML, 'report' for old text report.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel translation threads (workers).")
    parser.add_argument("--tpm", type=int, default=11000, help="Max tokens per minute rate limit to set for Groq.")
    args = parser.parse_args()

    # Set default extension based on format
    if args.ext is None:
        args.ext = ".xml" if args.format == "xml" else ".txt"

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please set it before running the script.")

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    rules_path = repo_root / "rules.md"
    
    print("Loading rules...")
    rules_content = load_rules(rules_path)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(list(input_dir.glob("*.xml")))
    if not xml_files:
        print(f"No XML files found in {input_dir}")
        return 1

    # Pre-filter files to process
    files_to_process = []
    skip_count = 0
    for xml_path in xml_files:
        output_path = output_dir / f"{xml_path.stem}{args.ext}"
        if output_path.exists() and not args.overwrite:
            skip_count += 1
        else:
            files_to_process.append(xml_path)

    if args.limit:
        files_to_process = files_to_process[:args.limit]

    print(f"Found {len(xml_files)} files. Skipped {skip_count} existing files. {len(files_to_process)} files to translate.")
    if not files_to_process:
        print("All files processed!")
        return 0

    rate_limiter = TokenRateLimiter(max_tokens_per_minute=args.tpm)

    def process_one_file(xml_path: Path) -> tuple[str, str]:
        output_path = output_dir / f"{xml_path.stem}{args.ext}"
        tweets = extract_tweets(xml_path)
        if not tweets:
            return xml_path.name, "empty"

        formatted_tweets = "\n".join(f"{i+1}. {tweet}" for i, tweet in enumerate(tweets))

        if args.format == "report":
            prompt = (
                f"Please translate and profile the following tweets for one author. "
                f"Adhere strictly to the rules provided in the system instructions. "
                f"CRITICAL: You MUST output the report with the exact English headers: '👤 Gender: ', '🧠 What kind of person + why: ', '🏷️ Interests — primary, secondary, other + why: ', '📝 Translated tweets: ', '🔍 Notes: '. Do not translate these headers to Myanmar under any circumstances.\n\n"
                f"Tweets:\n{formatted_tweets}"
            )
            try:
                translated_report = call_groq(api_key, args.model, rules_content, prompt, rate_limiter)
                output_path.write_text(translated_report, encoding="utf-8")
                return xml_path.name, "success"
            except Exception as e:
                return xml_path.name, str(e)
        else:
            # XML format - translate in chunks of 10 tweets and robustly parse JSON output client-side (avoiding strict server-side JSON mode limits/wrapping issues)
            chunk_size = 10
            translated_tweets = []
            
            chunks = [tweets[i:i + chunk_size] for i in range(0, len(tweets), chunk_size)]
            
            def extract_json(text: str) -> str:
                text = text.strip()
                if "```" in text:
                    first_idx = text.find("```")
                    newline_idx = text.find("\n", first_idx)
                    if newline_idx != -1:
                        start_idx = newline_idx + 1
                    else:
                        start_idx = first_idx + 3
                    second_idx = text.find("```", start_idx)
                    if second_idx != -1:
                        return text[start_idx:second_idx].strip()
                    else:
                        return text[start_idx:].strip()
                return text

            try:
                for chunk_idx, chunk in enumerate(chunks):
                    formatted_chunk = "\n".join(f"{i+1}. {tweet}" for i, tweet in enumerate(chunk))
                    system_instruction = (
                        "You are a professional translator translating English tweets to Myanmar for NLP author profiling research.\n"
                        "Translate the input list of tweets following these strict rules:\n\n"
                        "RULES:\n"
                        "1. Translate normal English sentences and words to natural Myanmar language.\n"
                        "2. Keep in English: Brand names, platform names (like YouTube, Twitter), movie/song/game/book titles, usernames (@mentions), hashtags (#hashtags), and rare/unknown country/city names.\n"
                        "3. Remove completely: All URLs and t.co links.\n"
                        "4. Tone & Style: Write like a real Myanmar Twitter user, not a robotic word-for-word translator. Keep the original emotional, sarcastic, or short tone.\n"
                        "5. Gender neutrality: Do not use gender-revealing first-person forms (such as ကျွန်တော်, ကျွန်မ, ကျွန်ုပ်). Use neutral pronouns (like ငါ, ငါတို့, မင်း) or neutral phrasing.\n"
                        "6. Output Format: You must output a JSON object with a single key 'translations' containing a list of exactly "
                        f"{len(chunk)} translated strings in the exact same order as the input tweets. Output ONLY the raw JSON. "
                        "Do NOT wrap the JSON in ```json or ``` or any markdown code blocks. Do NOT include any backticks. "
                        "Your response must start with '{' and end with '}'. Example:\n"
                        "{\n"
                        "  \"translations\": [\"တွစ်တာ စာသား ၁\", \"တွစ်တာ စာသား ၂\"]\n"
                        "}"
                    )
                    prompt = f"Tweets to translate:\n{formatted_chunk}"
                    
                    translated_chunk_str = call_groq(
                        api_key, 
                        args.model, 
                        system_instruction, 
                        prompt, 
                        rate_limiter
                    )
                    
                    import json
                    clean_json = extract_json(translated_chunk_str)
                    try:
                        data = json.loads(clean_json)
                    except Exception as je:
                        sys.stderr.write(f"\n[JSON Error in Chunk {chunk_idx}] error: {je}\nRaw LLM output:\n{translated_chunk_str}\n")
                        sys.stderr.flush()
                        raise je
                    chunk_translations = data.get("translations", [])
                    if not isinstance(chunk_translations, list) or len(chunk_translations) != len(chunk):
                        raise ValueError(f"LLM response translations list mismatch: expected {len(chunk)}, got {len(chunk_translations) if isinstance(chunk_translations, list) else 'not a list'}")
                    
                    translated_tweets.extend(chunk_translations)
                
                xml_lines = ['<author lang="my">', '\t<documents>']
                for tweet in translated_tweets:
                    clean_tweet = str(tweet).replace("]]>", "]]")
                    xml_lines.append(f'\t\t<document><![CDATA[{clean_tweet}]]></document>')
                xml_lines.extend(['\t</documents>', '</author>'])
                xml_content = "\n".join(xml_lines) + "\n"
                
                output_path.write_text(xml_content, encoding="utf-8")
                return xml_path.name, "success"
            except Exception as e:
                return xml_path.name, str(e)

    success_count = 0
    fail_count = 0

    print(f"Starting parallel Groq translation with {args.workers} workers (TPM Limit: {args.tpm})...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one_file, xml_path): xml_path for xml_path in files_to_process}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Translating files"):
            name, result = future.result()
            if result == "success":
                success_count += 1
            else:
                fail_count += 1
                sys.stderr.write(f"\nFailed to process {name}: {result}\n")
                sys.stderr.flush()

    print(f"\nProcessing complete! Success: {success_count}, Failed: {fail_count}, Skipped: {skip_count}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
