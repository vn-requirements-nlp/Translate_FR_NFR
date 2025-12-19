# translate_requirements.py
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a professional technical translator for Software Requirements (FR/NFR). "
    "Translate English software/system requirements into natural, fluent Vietnamese. "
    "Avoid word-for-word translation; keep scientific/technical accuracy. "
    "Do NOT add explanations, do NOT add numbering/bullets. "
    "Preserve meaning, units, constraints, and parentheses. "
    "Each input line must produce exactly one Vietnamese line."
)

USER_PROMPT_TEMPLATE = """Translate the following requirement lines to Vietnamese.

Rules:
- Return exactly N Vietnamese lines for N input lines (same order).
- Do not merge lines. Do not split a line into multiple lines.
- No numbering, no bullets, no extra commentary.
- If an input line has spelling/grammar issues, correct it before translating.
- Keep each translation as a single line.

Input lines (JSON array of strings):
{lines_json}
"""


def read_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text.splitlines()


def chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    chunks = []
    buf = []
    for s in items:
        buf.append(s)
        if len(buf) >= chunk_size:
            chunks.append(buf)
            buf = []
    if buf:
        chunks.append(buf)
    return chunks


def call_with_retry(fn, max_retries: int = 8):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception:
            if attempt == max_retries - 1:
                raise
            sleep_s = (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_s)


def translate_batch(client: OpenAI, model: str, lines: list[str]) -> list[str]:
    lines_json = json.dumps(lines, ensure_ascii=False)
    user_prompt = USER_PROMPT_TEMPLATE.format(lines_json=lines_json)

    def _do():
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "translation_batch",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "translations": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["translations"],
                    },
                    "strict": True,
                }
            },
        )

        # output_text should be a JSON string matching the schema
        raw = resp.output_text or ""
        data = json.loads(raw)
        out = data["translations"]

        if len(out) != len(lines):
            raise RuntimeError(f"Batch size mismatch: in={len(lines)} out={len(out)}")

        # Ensure single-line outputs
        out = [t.replace("\n", " ").strip() for t in out]
        return out

    return call_with_retry(_do)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True, help="Input .txt (1 requirement per line)")
    ap.add_argument("--out_txt", required=True, help="Output Vietnamese .txt")
    ap.add_argument("--model", default="gpt-4o-mini", help="Model name")
    ap.add_argument("--batch_size", type=int, default=120, help="Lines per API call")
    ap.add_argument("--resume", action="store_true", help="Resume if out_txt already has some lines")
    args = ap.parse_args()

    in_path = Path(args.in_txt)
    out_path = Path(args.out_txt)

    src_lines = read_lines(in_path)

    done = 0
    out_lines: list[str] = []
    if args.resume and out_path.exists():
        existing = read_lines(out_path)
        done = len(existing)
        out_lines = existing
        print(f"[resume] Already translated: {done} lines")

    remaining = src_lines[done:]
    chunks = chunk_list(remaining, args.batch_size)

    client = OpenAI()  # reads OPENAI_API_KEY from env (loaded by load_dotenv)

    for batch in chunks:
        if all(not x.strip() for x in batch):
            out_lines.extend(["" for _ in batch])
        else:
            translated = translate_batch(client, args.model, batch)
            out_lines.extend(translated)

        out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"Progress: {len(out_lines)}/{len(src_lines)} lines")

    print("Done.")


if __name__ == "__main__":
    main()
