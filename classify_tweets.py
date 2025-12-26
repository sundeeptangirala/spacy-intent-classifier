"""
End-to-end script:
- Load spaCy text classification model
- Read tweets from JSONL file
- Run classification
- Write classified_tweets.jsonl with predictions
"""

import json
import pathlib
from typing import Dict, Any, List, Tuple

import spacy

# ---- CONFIG ----

# Path to your trained spaCy model directory
MODEL_DIR = pathlib.Path("app/model")  # change if different

# Input tweets file (JSONL)
INPUT_TWEETS_FILE = pathlib.Path("data/tweets.jsonl")

# Output classified tweets file (JSONL)
OUTPUT_TWEETS_FILE = pathlib.Path("data/classified_tweets.jsonl")


# ---- HELPERS ----

def load_model():
    """Load the spaCy pipeline that includes a textcat component."""
    nlp = spacy.load(MODEL_DIR)
    if "textcat" not in nlp.pipe_names and "textcat_multilabel" not in nlp.pipe_names:
        raise ValueError(
            f"Loaded model at {MODEL_DIR} has no textcat or textcat_multilabel component. "
            f"Pipeline components: {nlp.pipe_names}"
        )
    return nlp


def read_tweets(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Read tweets from a JSONL file."""
    tweets: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tweets.append(json.loads(line))
    return tweets


def classify_texts(
    nlp,
    texts: List[str],
    top_k: int = 3
) -> List[List[Tuple[str, float]]]:
    """Run the text classification model on a list of texts."""
    docs = list(nlp.pipe(texts))
    all_preds: List[List[Tuple[str, float]]] = []
    for doc in docs:
        cats = doc.cats
        sorted_cats = sorted(cats.items(), key=lambda x: x[1], reverse=True)
        all_preds.append(sorted_cats[:top_k])
    return all_preds


def write_classified_tweets(
    tweets: List[Dict[str, Any]],
    all_predictions: List[List[Tuple[str, float]]],
    out_path: pathlib.Path
) -> None:
    """Write the classified tweets as JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf8") as out_f:
        for tweet, preds in zip(tweets, all_predictions):
            tweet_out = dict(tweet)
            tweet_out["predictions"] = [
                {"label": label, "score": float(score)} for label, score in preds
            ]
            out_f.write(json.dumps(tweet_out) + "\n")


# ---- MAIN ----

def main(
    model_dir: pathlib.Path = MODEL_DIR,
    input_file: pathlib.Path = INPUT_TWEETS_FILE,
    output_file: pathlib.Path = OUTPUT_TWEETS_FILE,
    top_k: int = 3,
) -> None:
    print(f"Loading model from: {model_dir}")
    nlp = spacy.load(model_dir)

    print(f"Reading tweets from: {input_file}")
    tweets = read_tweets(input_file)
    texts = [t["text"] for t in tweets]

    if not texts:
        print("No tweets found in input file.")
        return

    print(f"Classifying {len(texts)} tweets...")
    all_predictions = classify_texts(nlp, texts, top_k=top_k)

    print(f"Writing classified tweets to: {output_file}")
    write_classified_tweets(tweets, all_predictions, output_file)
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify tweets from a JSONL file with a spaCy textcat model."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODEL_DIR),
        help="Path to spaCy model directory (default: app/model)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_TWEETS_FILE),
        help="Input tweets JSONL file (default: data/tweets.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_TWEETS_FILE),
        help="Output classified tweets JSONL file (default: data/classified_tweets.jsonl)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top labels to keep per tweet",
    )

    args = parser.parse_args()

    main(
        model_dir=pathlib.Path(args.model_dir),
        input_file=pathlib.Path(args.input),
        output_file=pathlib.Path(args.output),
        top_k=args.top_k,
    )
