import json
import re
from typing import List, Tuple, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from .nlp_utils import ensure_nltk

class MLChatbot:
    def __init__(self):
        ensure_nltk()
        self.analysis_text = ""
        self.recommendation_text = ""
        self.sentences: List[str] = []
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.sentence_vectors = None
        self.trained = False

        # Sentiment keywords
        self.positive_words = set([
            "good", "excellent", "strong", "growth", "profitable", "positive",
            "buy", "outperform", "recommend", "opportunity", "bullish", "upside",
            "healthy", "solid", "robust", "attractive", "gaining", "increasing"
        ])

        self.negative_words = set([
            "bad", "poor", "weak", "decline", "loss", "negative", "sell",
            "underperform", "avoid", "risk", "bearish", "downside", "concerning",
            "deteriorating", "falling", "decreasing", "risky", "volatile"
        ])

    def train(self, analysis_text: str, recommendation_text: str) -> bool:
        self.analysis_text = analysis_text or ""
        self.recommendation_text = recommendation_text or ""
        full_text = f"{self.analysis_text}\n\n{self.recommendation_text}".strip()

        if not full_text:
            self.trained = False
            return False

        self.sentences = sent_tokenize(full_text)
        if len(self.sentences) == 0:
            self.trained = False
            return False

        self.sentence_vectors = self.vectorizer.fit_transform(self.sentences)
        self.trained = True
        return True

    def extract_sentiment(self, text: str) -> str:
        words = word_tokenize((text or "").lower())
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        if pos_count > neg_count:
            return "positive"
        if neg_count > pos_count:
            return "negative"
        return "neutral"

    def find_relevant_sentences(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if not self.trained or self.sentence_vectors is None:
            return []
        question_vector = self.vectorizer.transform([question])
        sims = cosine_similarity(question_vector, self.sentence_vectors).flatten()
        top_idx = sims.argsort()[-top_k:][::-1]
        return [(self.sentences[i], float(sims[i])) for i in top_idx if sims[i] > 0.1]

    def extract_key_metrics(self) -> Dict:
        metrics = {}
        price_pattern = r"\$?(\d+\.?\d*)\s*(USD|billion|million|lakh|crore)"
        pe_pattern = r"P/E.*?(\d+\.?\d+)"
        full_text = f"{self.analysis_text} {self.recommendation_text}"
        prices = re.findall(price_pattern, full_text)
        if prices:
            metrics["prices"] = prices[:5]
        pe_match = re.search(pe_pattern, full_text)
        if pe_match:
            metrics["pe_ratio"] = pe_match.group(1)
        return metrics

    def answer_question(self, question: str) -> str:
        if not self.trained:
            return "Model not trained yet. Please train the model first on an analysis."
        q = (question or "").lower()

        is_good_bad = any(w in q for w in ["good", "bad", "recommend", "should i buy", "worth"])
        is_metrics = any(w in q for w in ["metrics", "numbers", "data", "ratio"])

        relevant = self.find_relevant_sentences(question, top_k=3)
        if not relevant:
            return "I couldn't find relevant information in the analysis. Please try rephrasing your question."

        parts = []
        if is_good_bad:
            sent = self.extract_sentiment(self.recommendation_text)
            label = {"positive":"POSITIVE","negative":"NEGATIVE","neutral":"NEUTRAL"}.get(sent, "NEUTRAL")
            parts.append(f"Based on the analysis, the outlook is {label}.")

        if is_metrics:
            metrics = self.extract_key_metrics()
            if metrics:
                parts.append(f"Key metrics found: {json.dumps(metrics, indent=2)}")

        parts.append("\nRelevant information from the analysis:")
        for s, score in relevant:
            parts.append(f"• {s}")

        rec_lower = (self.recommendation_text or "").lower()
        if "buy" in rec_lower:
            parts.append("\n💡 The recommendation mentions BUY signal.")
        elif "sell" in rec_lower:
            parts.append("\n⚠️ The recommendation mentions SELL signal.")
        elif "hold" in rec_lower:
            parts.append("\n⏸️ The recommendation mentions HOLD signal.")

        return " ".join(parts)
