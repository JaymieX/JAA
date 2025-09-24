"""Juliet synthetic fine-tuning data generator driven by `SythDataGen`.

This script streams the raw Juliet dataset, converts each record into the
conversation shape expected by `SythDataGen`, and writes the synthetic outputs
back to the raw data directory as `juliet_sft.jsonl`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# Ensure the project root (two levels up) is on sys.path so we can import settings.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from settings import config  # type: ignore  # Local project import
from data_collections.syth_data_gen.syth_data_gen import ModelSize, SythDataGen

RawRecord = Tuple[int, str, str]
Messages = List[Dict[str, str]]

SYSTEM_PROMPTS = [
    {
        "role": "system",
        "content": (
            "You are a helpful cyber security programming assistant. Given a vulnerable code snippet, "
            "output only the corrected code. Do not include explanations, markdown, or extra text."
        ),
    },
    {
        "role": "system",
        "content": (
            "You are a security-focused coding assistant. When you receive insecure code, respond with the "
            "fixed code only. Omit explanations, extra formatting, and commentary."
        ),
    },
    {
        "role": "system",
        "content": (
            "You are a vulnerability remediation bot. Return solely the corrected source that resolves the issue. "
            "Exclude markdown, prose, and justification."
        ),
    },
]


def iter_juliet_raw(raw_path: Path) -> Iterator[RawRecord]:
    """Yield `(line_number, bad_text, good_text)` tuples from the Juliet raw file."""
    with raw_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:  # Skip malformed lines but keep going
                print(f"[warn] Skipping line {line_number}: invalid JSON ({exc})")
                continue

            bad  = payload.get("bad")
            good = payload.get("good")
            if not isinstance(bad, str) or not isinstance(good, str):
                print(f"[warn] Skipping line {line_number}: missing 'bad'/'good' strings")
                continue

            yield line_number, bad, good


def build_seed_messages(user_text: str, assistant_text: str) -> Messages:
    """Return a minimal seed conversation for SythDataGen."""
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def write_conversations(
    outfile: Path,
    conversations: Iterable[Dict[str, object]],
) -> None:
    """Append dictionaries as JSONL to `outfile`; caller ensures truncation if needed."""
    with outfile.open("a", encoding="utf-8") as handle:
        for conversation in conversations:
            handle.write(json.dumps(conversation, ensure_ascii=False) + "\n")


def generate_sft(
    variations: int,
    limit: Optional[int],
    rng_seed: Optional[int],
    log_every: int,
    include_seed: bool,
    size: str,
) -> None:
    """Run the full Juliet synthetic data generation pipeline."""
    raw_path    = config.get_raw_file_path("juliet_raw.jsonl")
    output_path = config.get_raw_file_path("juliet_sft.jsonl")
    output_path.unlink(missing_ok=True)  # Always overwrite the file

    if not raw_path.exists():
        print(f"[error] Juliet raw file not found: {raw_path}")
        return

    generator = SythDataGen(size=size)
    generator.set_system_prompts(SYSTEM_PROMPTS)

    seeds_processed = 0
    conversations_written = 0

    for line_number, bad_text, good_text in iter_juliet_raw(raw_path):
        if limit is not None and seeds_processed >= limit:
            break

        seed_messages = build_seed_messages(bad_text, good_text)
        seed_rng = rng_seed + seeds_processed if rng_seed is not None else None
        
        synthetic_conversations = generator.generate(
            seed_messages,
            variations=variations,
            rng_seed=seed_rng,
        )

        packaged: List[Dict[str, object]] = []
        for idx, convo in enumerate(synthetic_conversations, start=1):
            record: Dict[str, object] = {
                "seed_line": line_number,
                "variation": idx,
                "messages": convo,
            }
            
            if include_seed:
                record["seed"] = seed_messages
                
            packaged.append(record)

        if packaged:
            write_conversations(output_path, packaged)
            conversations_written += len(packaged)

        seeds_processed += 1
        if seeds_processed % log_every == 0:
            print(
                f"Processed {seeds_processed} seeds -> {conversations_written} synthetic conversations"
            )

    print(
        f"Done. Seeds processed: {seeds_processed}, synthetic conversations written: {conversations_written}"
    )
    
    print(f"Output saved to: {output_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Juliet SFT data using the shared SythDataGen pipeline.",
    )

    parser.add_argument(
        "-v", "--variations",
        type=int,
        default=1,
        help="Number of synthetic conversations to generate per raw seed (default: 1).",
    )

    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of raw seeds to process.",
    )

    parser.add_argument(
        "-r", "--rng", "--rng-seed",
        dest="rng_seed",
        type=int,
        default=42,
        help="Base RNG seed. Each seed increments by 1 to keep outputs deterministic.",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print a progress message after this many seeds (default: 100).",
    )

    parser.add_argument(
        "--include-seed",
        action="store_true",
        help="Include the original seed conversation alongside each synthetic output.",
    )

    parser.add_argument(
        "-s", "--size",
        type=lambda value: value.lower(),
        choices=[option.value for option in ModelSize],
        default=ModelSize.SMALL.value,
        help="Model profile to use: 'small' loads the 3B 4-bit model, 'large' loads the 7B model.",
    )

    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.variations <= 0:
        print("[warn] No variations requested; nothing to do.")
        return
    if args.log_every <= 0:
        print("[warn] log_every must be positive; defaulting to 100.")
        args.log_every = 100

    generate_sft(
        variations=args.variations,
        limit=args.limit,
        rng_seed=args.rng_seed,
        log_every=args.log_every,
        include_seed=args.include_seed,
        size=args.size,
    )


if __name__ == "__main__":
    main()

