# utils.py
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
MAX_TOKENS   = 512
CHUNK_STRIDE = 50    # overlap between windows (tokens)


def chunk_text(text: str) -> list[str]:
    """
    Split a long text into overlapping chunks of MAX_TOKENS tokens.
    Uses the FinBERT tokenizer to measure real token counts.
    Returns a list of decoded string chunks.
    """
    tokens = TOKENIZER.encode(text, add_special_tokens=False)

    if len(tokens) <= MAX_TOKENS - 2:   # fits in one pass (2 reserved for [CLS]/[SEP])
        return [text]

    chunks = []
    step   = MAX_TOKENS - 2 - CHUNK_STRIDE
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + MAX_TOKENS - 2]
        chunks.append(TOKENIZER.decode(chunk_tokens, skip_special_tokens=True))
        if start + MAX_TOKENS - 2 >= len(tokens):
            break
    return chunks


def run_inference(pipe, articles: list[dict]) -> list[dict]:
    """
    Run FinBERT inference over a list of article dicts.
    Each article gets a 'sentiment', 'score', and per-label scores added to it.
    Handles chunking transparently.
    """
    results = []
    for article in articles:
        text   = article["text"]
        chunks = chunk_text(text)

        # Accumulate label scores across all chunks
        label_totals = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for chunk in chunks:
            predictions = pipe(chunk)[0]   # list of {label, score}
            for pred in predictions:
                label_lower = pred["label"].lower()
                if label_lower in label_totals:
                    label_totals[label_lower] += pred["score"]

        # Average across chunks
        n = len(chunks)
        avg_scores = {k: v / n for k, v in label_totals.items()}

        # The winning label
        best_label = max(avg_scores, key=lambda k: avg_scores[k])
        best_score = avg_scores[best_label]

        results.append({
            **article,
            "sentiment":        best_label,
            "confidence":       round(best_score, 4),
            "positive_score":   round(avg_scores["positive"], 4),
            "negative_score":   round(avg_scores["negative"], 4),
            "neutral_score":    round(avg_scores["neutral"],  4),
        })
    return results


def aggregate_sentiment(results: list[dict]) -> dict:
    """
    Compute summary stats from a batch of inference results.
    """
    if not results:
        return {}

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    total_positive = total_negative = total_neutral = 0.0

    for r in results:
        counts[r["sentiment"]] += 1
        total_positive += r["positive_score"]
        total_negative += r["negative_score"]
        total_neutral  += r["neutral_score"]

    n = len(results)
    # Composite score: net sentiment from -1 (fully negative) to +1 (fully positive)
    composite = round((total_positive - total_negative) / n, 4)

    return {
        "total_articles":    n,
        "positive_count":    counts["positive"],
        "negative_count":    counts["negative"],
        "neutral_count":     counts["neutral"],
        "avg_positive":      round(total_positive / n, 4),
        "avg_negative":      round(total_negative / n, 4),
        "avg_neutral":       round(total_neutral  / n, 4),
        "composite_score":   composite,
        "overall_sentiment": "positive" if composite > 0.1
                             else "negative" if composite < -0.1
                             else "neutral",
    }