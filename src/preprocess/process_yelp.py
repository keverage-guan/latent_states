#!/usr/bin/env python3
"""
process_yelp.py

Extract the fields needed for sentiment analysis from the Yelp dataset.

Input:
    assets/raw/yelp_academic_dataset_review.json

Output:
    data/yelp_sentiment.jsonl

Each output line contains:
{
    "id": "...",
    "timestamp": "...",
    "sentiment": <star rating>,
    "text": "..."
}
"""

from pathlib import Path
import json

RAW_DIR = Path("assets/raw")
OUT_DIR = Path("data")

INPUT_FILE = RAW_DIR / "yelp_academic_dataset_review.json"
OUTPUT_FILE = OUT_DIR / "yelp_sentiment.jsonl"

OUT_DIR.mkdir(parents=True, exist_ok=True)

count = 0
skipped_missing = 0
skipped_duplicate = 0
seen_ids = set()

with INPUT_FILE.open("r", encoding="utf-8") as fin, \
     OUTPUT_FILE.open("w", encoding="utf-8") as fout:

    for line in fin:
        review = json.loads(line)

        # Pull the requisite fields; any missing one disqualifies the row.
        review_id = review.get("review_id")
        timestamp = review.get("date")
        sentiment = review.get("stars")
        text = review.get("text", "").strip()

        if not all([review_id, timestamp, sentiment is not None, text]):
            skipped_missing += 1
            continue

        # Drop reviews whose id was already written.
        if review_id in seen_ids:
            skipped_duplicate += 1
            continue
        seen_ids.add(review_id)

        processed = {
            "id": review_id,
            "timestamp": timestamp,
            "sentiment": sentiment,
            "text": text,
        }

        fout.write(json.dumps(processed, ensure_ascii=False) + "\n")
        count += 1

print(f"Processed {count:,} reviews.")
print(f"Skipped {skipped_missing:,} reviews with missing values.")
print(f"Skipped {skipped_duplicate:,} duplicate reviews.")
print(f"Saved to {OUTPUT_FILE}")