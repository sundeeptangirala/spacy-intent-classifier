"""
spaCy Multi-Task Classifier: Intent + Sentiment Analysis

Using spaCy's textcat_multilabel component to simultaneously predict:
1. Intent categories (card_declined, account_balance, etc.)
2. Sentiment with attrition risk (positive, neutral_atrisk, negative)

Requires a trained spaCy model with textcat_multilabel component.
"""

import json
import pathlib
from typing import Dict, List, Any
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# ---- CONFIG ----

MODEL_DIR = pathlib.Path("app/model")  # Trained spaCy model
INPUT_FILE = pathlib.Path("data/tweets.jsonl")
OUTPUT_FILE = pathlib.Path("data/classified_with_sentiment.jsonl")

# Attrition risk keywords
ATRISK_KEYWORDS = [
    "cancel", "closing account", "switching", "leaving", "done with",
    "terrible", "awful", "worst", "disgusted", "disappointed",
    "never again", "lost customer", "unacceptable", "horrible"
]

# ---- LOAD MODEL ----

def load_spacy_model():
    """Load trained spaCy model with textcat component."""
    nlp = spacy.load(MODEL_DIR)
    
    # Add sentiment analyzer using TextBlob
    if "spacytextblob" not in nlp.pipe_names:
        nlp.add_pipe("spacytextblob")
    
    return nlp

# ---- CLASSIFICATION ----

def classify_intent_and_sentiment(text: str, nlp) -> Dict[str, Any]:
    """
    Classify both intent and sentiment using spaCy.
    
    Returns:
        Dictionary with intent predictions and sentiment analysis
    """
    doc = nlp(text)
    
    # Get intent predictions from textcat
    intent_cats = {}
    if doc.cats:
        # Sort by score
        sorted_cats = sorted(doc.cats.items(), key=lambda x: x[1], reverse=True)
        intent_cats = {
            "top_intent": sorted_cats[0][0],
            "confidence": sorted_cats[0][1],
            "all_scores": dict(sorted_cats[:3])  # Top 3
        }
    
    # Get sentiment using TextBlob
    polarity = doc._.blob.polarity  # -1 (negative) to 1 (positive)
    subjectivity = doc._.blob.subjectivity  # 0 (objective) to 1 (subjective)
    
    # Determine sentiment category with attrition risk
    sentiment_result = analyze_attrition_risk(text, polarity, subjectivity)
    
    return {
        "intent": intent_cats,
        "sentiment": sentiment_result
    }

def analyze_attrition_risk(text: str, polarity: float, subjectivity: float) -> Dict[str, Any]:
    """
    Analyze sentiment with customer attrition risk.
    
    Categories:
    - positive: polarity > 0.3
    - neutral_atrisk: polarity between -0.3 and 0.3, with risk keywords
    - negative: polarity < -0.3 or has strong attrition keywords
    """
    text_lower = text.lower()
    risk_keywords_found = [kw for kw in ATRISK_KEYWORDS if kw in text_lower]
    
    # Determine sentiment and risk
    if polarity > 0.3 and not risk_keywords_found:
        sentiment = "positive"
        attrition_risk = "low"
    elif polarity < -0.5 or len(risk_keywords_found) >= 2:
        sentiment = "negative"
        attrition_risk = "high"
    elif risk_keywords_found:
        sentiment = "neutral_atrisk"
        attrition_risk = "medium" if len(risk_keywords_found) == 1 else "high"
    elif polarity < -0.1:
        sentiment = "neutral_atrisk"
        attrition_risk = "medium"
    else:
        sentiment = "neutral_atrisk"
        attrition_risk = "low"
    
    return {
        "category": sentiment,
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "attrition_risk": attrition_risk,
        "risk_keywords": risk_keywords_found[:3]  # Top 3
    }

# ---- PROCESSING ----

def process_tweets_with_spacy(input_path: pathlib.Path, output_path: pathlib.Path):
    """Process tweets using spaCy for intent + sentiment."""
    print("Loading spaCy model...")
    nlp = load_spacy_model()
    
    print(f"Reading tweets from {input_path}...")
    tweets = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tweets.append(json.loads(line))
    
    print(f"Processing {len(tweets)} tweets...")
    results = []
    
    for idx, tweet in enumerate(tweets, 1):
        text = tweet["text"]
        
        # Classify
        classification = classify_intent_and_sentiment(text, nlp)
        
        # Combine results
        result = {
            "id": tweet.get("id", f"tweet_{idx}"),
            "user": tweet.get("user", "unknown"),
            "text": text,
            "intent_category": classification["intent"].get("top_intent", "unknown"),
            "intent_confidence": round(classification["intent"].get("confidence", 0), 3),
            "sentiment": classification["sentiment"]["category"],
            "sentiment_polarity": classification["sentiment"]["polarity"],
            "attrition_risk": classification["sentiment"]["attrition_risk"],
            "risk_keywords": "|".join(classification["sentiment"]["risk_keywords"]),
            "priority": determine_priority(classification)
        }
        
        results.append(result)
        
        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(tweets)} tweets")
    
    # Write output
    print(f"Writing results to {output_path}...")
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"Done! Processed {len(results)} tweets.")
    print_summary(results)

def determine_priority(classification: Dict) -> str:
    """Determine ticket priority based on intent and sentiment."""
    intent = classification["intent"].get("top_intent", "")
    risk = classification["sentiment"]["attrition_risk"]
    
    # High priority: fraud or high attrition risk
    if "fraud" in intent or risk == "high":
        return "HIGH"
    elif risk == "medium" or "card" in intent or "atm" in intent:
        return "MEDIUM"
    else:
        return "LOW"

def print_summary(results: List[Dict]):
    """Print summary statistics."""
    total = len(results)
    
    # Sentiment breakdown
    sentiments = {}
    for r in results:
        s = r["sentiment"]
        sentiments[s] = sentiments.get(s, 0) + 1
    
    # Risk breakdown
    risks = {}
    for r in results:
        risk = r["attrition_risk"]
        risks[risk] = risks.get(risk, 0) + 1
    
    print("\n=== SUMMARY ===")
    print(f"Total tweets: {total}")
    print("\nSentiment breakdown:")
    for sent, count in sentiments.items():
        pct = (count/total)*100
        print(f"  {sent}: {count} ({pct:.1f}%)")
    
    print("\nAttrition risk:")
    for risk, count in risks.items():
        pct = (count/total)*100
        print(f"  {risk}: {count} ({pct:.1f}%)")

# ---- MAIN ----

if __name__ == "__main__":
    import sys
    
    input_file = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_FILE
    output_file = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_FILE
    
    process_tweets_with_spacy(input_file, output_file)
