#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from functools import lru_cache
from pathlib import Path


URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
DOC_RE = re.compile(r"<document><!\[CDATA\[(.*?)\]\]></document>", re.S)
MARKER_RE = re.compile(r"^@@(\d+)@@$")


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = URL_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def protect_tokens(text: str) -> tuple[str, dict[str, str]]:
    tokens: dict[str, str] = {}
    counter = 0

    def replace(match: re.Match[str], kind: str) -> str:
        nonlocal counter
        token = f"PAN18_{kind}_{counter}_PAN18"
        tokens[token] = match.group(0)
        counter += 1
        return token

    text = MENTION_RE.sub(lambda m: replace(m, "MENTION"), text)
    text = HASHTAG_RE.sub(lambda m: replace(m, "HASH"), text)
    return text, tokens


def translate_text(text: str, retries: int = 1, retry_wait: float = 5.0) -> str:
    last_error: Exception | None = None
    attempt = 0
    while True:
        try:
            data = urllib.parse.urlencode(
                {
                    "client": "gtx",
                    "sl": "en",
                    "tl": "my",
                    "dt": "t",
                    "q": text,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                "https://translate.googleapis.com/translate_a/single",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            raw = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
            payload = json.loads(raw)
            return "".join(part[0] for part in payload[0] if part and part[0])
        except HTTPError as exc:
            last_error = exc
            attempt += 1
            if exc.code == 429:
                time.sleep(min(300.0, retry_wait * attempt))
                continue
        except Exception as exc:  # pragma: no cover - network/remote error handling
            last_error = exc
            attempt += 1
            time.sleep(min(300.0, retry_wait * attempt))

        if attempt >= retries:
            raise RuntimeError(f"translation failed after {retries} attempts: {last_error}")


@lru_cache(maxsize=1)
def load_local_translator():
    import torch
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def translate_local_docs(docs: list[str], batch_size: int = 8) -> list[str]:
    import torch

    tokenizer, model = load_local_translator()
    tokenizer.src_lang = "en"
    translated_docs: list[str] = []

    for start in range(0, len(docs), batch_size):
        chunk = docs[start : start + batch_size]
        prepared: list[str] = []
        token_maps: list[dict[str, str]] = []
        for doc in chunk:
            protected, token_map = protect_tokens(doc)
            prepared.append(protected)
            token_maps.append(token_map)

        encoded = tokenizer(prepared, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id("my"),
                max_new_tokens=96,
                num_beams=1,
            )

        outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for output, token_map in zip(outputs, token_maps):
            for token, original in token_map.items():
                output = output.replace(token, original)
            translated_docs.append(output.strip())

    return translated_docs


def split_translated(text: str, count: int) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    markers_seen = 0

    for line in text.splitlines():
        if MARKER_RE.fullmatch(line.strip()):
            parts.append("\n".join(current).strip())
            current = []
            markers_seen += 1
            continue
        current.append(line)

    parts.append("\n".join(current).strip())

    # Remove the leading empty chunk if the translation began with a marker-free prefix.
    if len(parts) == count + 1 and not parts[0]:
        parts = parts[1:]

    if markers_seen < count - 1 and len(parts) < count:
        # If the translation engine collapsed the markers, fall back to the raw body.
        return [text.strip()]

    return parts


def write_xml(src_text: str, translated_docs: list[str]) -> str:
    lines = ['<author lang="my">', "\t<documents>"]
    for doc in translated_docs:
        doc = doc.replace("]]>", "]]]]><![CDATA[>")
        lines.append(f"\t\t<document><![CDATA[{doc}]]></document>")
    lines.append("\t</documents>")
    lines.append("</author>")
    return "\n".join(lines) + "\n"


def clean_pronouns(text: str) -> str:
    text = text.replace("ကျွန်တော်တို့", "ငါတို့")
    text = text.replace("ကျွန်မတို့", "ငါတို့")
    text = text.replace("ကျွန်တော့်ရဲ့", "ငါ့")
    text = text.replace("ကျွန်မရဲ့", "ငါ့")
    text = text.replace("ကျွန်တော့်", "ငါ့")
    text = text.replace("ကျွန်တော်", "ငါ")
    text = text.replace("ကျွန်မ", "ငါ")
    text = text.replace("ကျွန်ုပ်", "ငါ")
    text = text.replace("သင်သည်", "မင်းသည်")
    text = text.replace("သင့်ကို", "မင်းကို")
    text = text.replace("သင့်ရဲ့", "မင်းရဲ့")
    text = text.replace("သင့်", "မင်း")
    text = text.replace("သင် ", "မင်း ")
    return text


def process_file(src_path: Path, dst_path: Path, retry_wait: float, backend: str) -> None:
    src_text = src_path.read_text(encoding="utf-8")
    docs = DOC_RE.findall(src_text)
    cleaned = [clean_text(doc) for doc in docs]
    if backend == "local":
        translated_docs = translate_local_docs(cleaned)
    else:
        combined: list[str] = []
        for i, doc in enumerate(cleaned):
            combined.append(doc)
            if i != len(cleaned) - 1:
                combined.append(f"@@{i}@@")
        try:
            translated = translate_text("\n".join(combined), retry_wait=retry_wait)
            translated_docs = split_translated(translated, len(cleaned))
            if len(translated_docs) != len(cleaned):
                raise RuntimeError(
                    f"{src_path.name}: translated doc count mismatch "
                    f"({len(translated_docs)} != {len(cleaned)})"
                )
        except Exception:
            translated_docs = translate_local_docs(cleaned)

    translated_docs = [clean_pronouns(doc) for doc in translated_docs]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(write_xml(src_text, translated_docs), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate PAN18 XML files to Myanmar.")
    parser.add_argument("--input-dir", default="en/text")
    parser.add_argument("--output-dir", default="my/text")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--retry-wait", type=float, default=20.0)
    parser.add_argument("--backend", choices=["local", "remote"], default="local")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob("*.xml"))
    batch = files[args.start : args.start + args.count]

    if not batch:
        print("No files selected.")
        return 0

    for index, src_path in enumerate(batch, start=args.start + 1):
        dst_path = output_dir / src_path.name
        if dst_path.exists() and not args.force:
            print(f"SKIP {index}/{len(files)} {src_path.name}")
            continue
        print(f"TRANSLATE {index}/{len(files)} {src_path.name}")
        if args.backend == "local":
            process_file(src_path, dst_path, args.retry_wait, "local")
        else:
            process_file(src_path, dst_path, args.retry_wait, "remote")
        print(f"WRITE {dst_path}")
        time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
