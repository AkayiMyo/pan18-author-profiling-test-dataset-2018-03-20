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

def call_gemini(api_key: str, model: str, system_instruction: str, prompt: str, response_mime_type: str = None) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    generation_config = {"temperature": 0.2}
    if response_mime_type:
        generation_config["responseMimeType"] = response_mime_type

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "systemInstruction": {
            "parts": [
                {"text": system_instruction}
            ]
        },
        "generationConfig": generation_config
    }
    
    max_retries = 8
    for attempt in range(max_retries):
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=180.0)
            response.raise_for_status()
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # Print token stats and cost estimate live
            usage = result.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            if "pro" in model.lower():
                input_cost = (prompt_tokens / 1000000) * 1.25
                output_cost = (output_tokens / 1000000) * 5.00
            else:
                input_cost = (prompt_tokens / 1000000) * 0.075
                output_cost = (output_tokens / 1000000) * 0.30
            cost = input_cost + output_cost
            sys.stdout.write(f"\n[{model}] Tokens: {prompt_tokens} in / {output_tokens} out | Est. Cost: ${cost:.6f} USD\n")
            sys.stdout.flush()
            
            return text
        except httpx.HTTPStatusError as e:
            if (e.response.status_code == 429 or 500 <= e.response.status_code < 600) and attempt < max_retries - 1:
                wait_time = 30.0 * (attempt + 1) if e.response.status_code == 429 else 10.0 * (attempt + 1)
                sys.stderr.write(f"\n[HTTP Error {e.response.status_code}] Details: {e.response.text}\nWaiting {wait_time}s before retry...\n")
                sys.stderr.flush()
                time.sleep(wait_time)
                continue
            raise e
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 10.0 * (attempt + 1)
                sys.stderr.write(f"\n[Error: {e}] Waiting {wait_time}s before retry...\n")
                sys.stderr.flush()
                time.sleep(wait_time)
                continue
            raise e
    raise RuntimeError("Failed to call Gemini API after retries.")

def main() -> int:
    parser = argparse.ArgumentParser(description="Translate PAN18 English XML files to Myanmar using Gemini API.")
    parser.add_argument("--input-dir", default="en/text", help="Directory containing English XML files.")
    parser.add_argument("--output-dir", default="my/text", help="Directory to save Myanmar translated files.")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini API model name.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of files to process.")
    parser.add_argument("--ext", default=None, help="Extension of the output files (e.g., .txt or .xml). If None, defaults to match the format.")
    parser.add_argument("--format", choices=["xml", "report"], default="xml", help="Output format: 'xml' for translated tweets XML, 'report' for old text report.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel translation threads (workers).")
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay in seconds between starting parallel files to distribute load.")
    args = parser.parse_args()

    # Set default extension based on format
    if args.ext is None:
        args.ext = ".xml" if args.format == "xml" else ".txt"

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        return 1

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

    # Pre-filter files to process to allow accurate tqdm count
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
                f"For the '🔍 Notes' section, focus on explaining key translation decisions (such as names, platforms, brands, country names, mentions, hashtags, or transliterated words) rather than explaining every common word, to keep it concise and high-quality.\n\n"
                f"Tweets:\n{formatted_tweets}"
            )
            try:
                translated_report = call_gemini(api_key, args.model, rules_content, prompt)
                output_path.write_text(translated_report, encoding="utf-8")
                return xml_path.name, "success"
            except Exception as e:
                return xml_path.name, str(e)
        else:
            # XML format - ask LLM for JSON list of translations to ensure robust XML generation
            system_instruction = (
                "You are a professional translator translating English tweets to Myanmar for NLP author profiling research.\n"
                "Translate the input list of tweets following these strict rules:\n\n"
                "RULES:\n"
                "1. Translate normal English sentences and words to natural Myanmar language.\n"
                "2. Keep in English: Brand names, platform names (like YouTube, Twitter), movie/song/game/book titles, usernames (@mentions), hashtags (#hashtags), and rare/unknown country/city names.\n"
                "3. Remove completely: All URLs and t.co links.\n"
                "4. Tone & Style: Write like a real Myanmar Twitter user, not a robotic word-for-word translator. Keep the original emotional, sarcastic, or short tone.\n"
                "5. Gender neutrality: Do not use gender-revealing first-person forms (such as ကျွန်တော်, ကျွန်မ, ကျွန်ုပ်). Use neutral pronouns (like ငါ, ငါတို့, မင်း) or neutral phrasing.\n"
                "6. Output Format: You must output the translations as a raw JSON list of strings, matching the exact order of the input tweets. Output nothing else. Do not wrap the JSON in ```json or markdown blocks. Just output the raw JSON list of strings. Example: [\"တွစ်တာ စာသား ၁\", \"တွစ်တာ စာသား ၂\"]"
            )
            prompt = f"Tweets to translate:\n{formatted_tweets}"
            try:
                translated_report = call_gemini(api_key, args.model, system_instruction, prompt, response_mime_type="application/json")
                
                clean_json = translated_report.strip()
                if clean_json.startswith("```"):
                    lines = clean_json.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].startswith("```"):
                        lines = lines[:-1]
                    clean_json = "\n".join(lines).strip()
                
                import json
                translated_tweets = json.loads(clean_json)
                if not isinstance(translated_tweets, list):
                    raise ValueError("LLM response is not a JSON list")
                
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

    print(f"Starting parallel translation with {args.workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i, xml_path in enumerate(files_to_process):
            futures.append(executor.submit(process_one_file, xml_path))
            if args.sleep > 0 and i < len(files_to_process) - 1:
                time.sleep(args.sleep)
        
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
