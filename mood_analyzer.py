# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Optional, Tuple

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        # Keep words (including contractions) and discard surrounding
        # punctuation so that "happy!" matches the word "happy".
        return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words add one point and negative words subtract one point.
        The negation rule reverses the next sentiment-bearing word after
        "not", "never", "no", or the colloquial negator "ain't".
        """
        score, _, _, _ = self._analyze(text)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        If both positive and negative evidence appear, return "mixed".
        Otherwise, a positive score returns "positive", a negative score
        returns "negative", and a zero score returns "neutral".
        """
        score, positive_hits, negative_hits, _ = self._analyze(text)

        # Evidence of both polarities is more informative than the net score:
        # it indicates mixed sentiment even when one side is stronger.
        if positive_hits and negative_hits:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Shared analysis and explanations
    # ---------------------------------------------------------------------

    def _analyze(self, text: str) -> Tuple[int, List[str], List[str], List[str]]:
        """Return the score, polarity matches, and tokens for one text.

        A simple negation word ("not", "never", "no", or "ain't") reverses
        the polarity of the next sentiment-bearing word. The negation is then
        consumed, so it does not affect later words in the sentence.
        """
        tokens = self.preprocess(text)
        negators = {"not", "never", "no", "ain't"}
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0
        negate_next = False

        for token in tokens:
            if token in negators:
                negate_next = True
                continue

            if token in self.positive_words:
                if negate_next:
                    negative_hits.append(token)
                    score -= 1
                else:
                    positive_hits.append(token)
                    score += 1
                negate_next = False
            elif token in self.negative_words:
                if negate_next:
                    positive_hits.append(token)
                    score += 1
                else:
                    negative_hits.append(token)
                    score -= 1
                negate_next = False

        return score, positive_hits, negative_hits, tokens

    # ---------------------------------------------------------------------
    # Explanations
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """Explain the matched words, negation effects, score, and label."""
        score, positive_hits, negative_hits, tokens = self._analyze(text)
        label = self.predict_label(text)
        return (
            f"Tokens = {tokens}; score = {score}; label = {label}; "
            f"positive evidence = {positive_hits or '[]'}; "
            f"negative evidence = {negative_hits or '[]'}"
        )
