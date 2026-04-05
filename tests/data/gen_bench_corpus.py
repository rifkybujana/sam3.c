#!/usr/bin/env python3
"""
gen_bench_corpus.py - Generate a deterministic corpus of CLIP-style captions
                      for the tokenizer throughput benchmark.

Usage:
    python3 tests/data/gen_bench_corpus.py                  # default: 10000 lines, seed 0
    python3 tests/data/gen_bench_corpus.py --count 1000     # custom size
    python3 tests/data/gen_bench_corpus.py --seed 42        # custom seed

Writes to tests/data/bench_corpus.txt (one caption per line, LF-terminated).
Output is deterministic for a given (count, seed).
"""

import argparse
import random
from pathlib import Path

ARTICLES = ["a/an", "the", "a/an"]  # weight indefinite articles 2x

ADJECTIVES = [
    "small", "large", "bright", "dark", "red", "blue", "green", "yellow",
    "orange", "purple", "black", "white", "golden", "silver", "tiny", "huge",
    "beautiful", "elegant", "rustic", "modern", "vintage", "shiny", "dusty",
    "old", "new", "wooden", "metal", "glass", "stone", "plastic",
]

NOUNS = [
    "cat", "dog", "bird", "horse", "rabbit", "elephant", "tiger", "bear",
    "car", "bicycle", "boat", "plane", "train", "bus", "truck", "motorcycle",
    "house", "building", "tower", "bridge", "church", "castle", "barn",
    "tree", "flower", "mountain", "river", "lake", "beach", "forest",
    "cup", "bowl", "plate", "book", "lamp", "chair", "table", "clock",
    "phone", "camera", "guitar", "piano", "painting", "sculpture", "statue",
    "person", "child", "woman", "man", "dancer", "musician", "chef",
]

PREPOSITIONS = ["on", "beside", "near", "in front of", "behind", "under", "above"]

CONTEXTS = [
    "a wooden table", "the grass", "a sandy beach", "a snowy field",
    "the street", "a stone wall", "a park bench", "the forest floor",
    "a city sidewalk", "a countryside road", "a marble floor", "the water",
    "a kitchen counter", "a bed of leaves", "a rocky cliff", "a garden path",
    "an old bookshelf", "a moonlit lake", "the sunset sky", "a cobblestone street",
]

TEMPLATES = [
    "{article} photo of {article} {adj} {noun}",
    "{article} {adj} {noun} {prep} {context}",
    "a photograph of {article} {adj} {noun}",
    "{article} {noun} {prep} {context}",
    "a detailed photo of {article} {adj} {noun} {prep} {context}",
    "{article} high quality image of {article} {adj} {noun}",
    "a picture showing {article} {adj} {noun}",
    "{article} {adj} {adj2} {noun} {prep} {context}",
    "a close up of {article} {adj} {noun}",
    "{article} {noun} and {article} {adj} {noun}",
]


def generate(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        template = rng.choice(TEMPLATES)
        caption = template.format(
            article=rng.choice(ARTICLES),
            adj=rng.choice(ADJECTIVES),
            adj2=rng.choice(ADJECTIVES),
            noun=rng.choice(NOUNS),
            prep=rng.choice(PREPOSITIONS),
            context=rng.choice(CONTEXTS),
        )
        out.append(fix_articles(caption))
    return out


def fix_articles(caption: str) -> str:
    """Resolve 'a/an' sentinel tokens to the correct English article."""
    words = caption.split(" ")
    for i, w in enumerate(words):
        if w == "a/an":
            nxt = words[i + 1] if i + 1 < len(words) else ""
            words[i] = "an" if nxt and nxt[0].lower() in "aeiou" else "a"
    return " ".join(words)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "bench_corpus.txt",
    )
    args = parser.parse_args()

    lines = generate(args.count, args.seed)
    text = "\n".join(lines) + ("\n" if lines else "")
    args.output.write_text(text, encoding="utf-8")
    size = args.output.stat().st_size
    print(f"Wrote {args.count} lines ({size} bytes) to {args.output}")


if __name__ == "__main__":
    main()
