"""
Recognizing which ready-made question a plain-language question asks.

A question is compared with each ready-made question's wordings: the label of its button
and its query read back as English. A wording is scored by how much of it the question
covers, each word weighed by how well it tells the wordings apart, so the framing all
questions share decides little and the words only one of them uses decide most.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from typing_extensions import Dict, FrozenSet, List, Optional, Tuple

from cramera_protocol.questions import QuestionMatch, Verbalization
from cramera_bridge.questions.presets import Preset

MINIMUM_SIMILARITY = 70.0
"""
The similarity, out of 100, below which no ready-made question counts as asked.
"""

FULL_SIMILARITY = 100.0
"""
The similarity of a question covering every word of a wording.
"""

UNMATCHED_QUESTION_REPLY = "Sorry, I cannot answer that question."
"""
What a question no ready-made question matches is answered with.
"""

UNHEARD_WORD_FACTOR = 2.0
"""
How many wordings' worth a word no wording uses counts for: more than any word they do
use, since a question saying it asks about something none of them is about.
"""

NON_WORD_CHARACTERS = re.compile(r"[\W_]+")
"""
What separates words: everything but letters and digits.
"""

# %% words


@dataclass(frozen=True)
class SpokenText:
    """
    A question or a wording, read as words.
    """

    text: str
    """
    The text as written.
    """

    def words(self) -> List[str]:
        """
        :return: The lower-cased words in the order they were written.
        """
        return NON_WORD_CHARACTERS.sub(" ", self.text.lower()).split()

    def spellings(self) -> FrozenSet[str]:
        """
        :return: The words, plus each pair of neighbouring words written as one, so
            "pickup" is recognized as "pick up".
        """
        words = self.words()
        return frozenset(words) | {
            first + second for first, second in zip(words, words[1:])
        }


# %% matching


@dataclass(frozen=True)
class WordedPreset:
    """
    A ready-made question with its query read back as English, when it was.
    """

    preset: Preset
    """
    The ready-made question.
    """

    verbalization: Optional[Verbalization] = None
    """
    Its query read back as English.
    """

    def wordings(self) -> List[str]:
        """
        :return: The label of its button and, when worded, its query's sentence.
        """
        wordings = [self.preset.text]
        if self.verbalization is not None:
            wordings.append(self.verbalization.text)
        return wordings


@dataclass
class QuestionMatcher:
    """
    Recognizes which of the ready-made questions on offer a question asks.
    """

    presets: List[WordedPreset]
    """
    The ready-made questions on offer.
    """

    minimum_similarity: float = MINIMUM_SIMILARITY
    """
    The similarity below which a question counts as unanswerable.
    """

    weights: Dict[str, float] = field(init=False)
    """
    What each word of the wordings is worth: the number of wordings over the number
    using it.
    """

    def __post_init__(self) -> None:
        wordings = [set(SpokenText(wording).words()) for wording in self.all_wordings()]
        used_by: Dict[str, float] = {}
        for wording in wordings:
            for word in wording:
                used_by[word] = used_by.get(word, 0.0) + 1.0
        self.weights = {word: len(wordings) / count for word, count in used_by.items()}

    def all_wordings(self) -> List[str]:
        """
        :return: Every wording of every ready-made question.
        """
        return [wording for worded in self.presets for wording in worded.wordings()]

    def weight(self, word: str) -> float:
        """
        :param word: A word.
        :return: What it is worth in deciding which wording is meant.
        """
        return self.weights.get(word, UNHEARD_WORD_FACTOR * len(self.all_wordings()))

    def match(self, text: str) -> QuestionMatch:
        """
        Prefer a question repeating a wording word for word, then the most similar.

        :param text: The question as asked.
        :return: The matching ready-made question, or the reply that none matches.
        """
        asked = SpokenText(text)
        said = asked.words()
        exact = next(
            (
                worded
                for worded in self.presets
                if said
                and any(said == SpokenText(one).words() for one in worded.wordings())
            ),
            None,
        )
        if exact is not None:
            return self.result(exact, self.comparison(asked, exact)[0])
        best: Optional[WordedPreset] = None
        best_pair = (0.0, -FULL_SIMILARITY)
        for worded in self.presets:
            pair = self.comparison(asked, worded)
            if pair > best_pair:
                best, best_pair = worded, pair
        return self.result(best, best_pair[0])

    def result(
        self, worded: Optional[WordedPreset], similarity: float
    ) -> QuestionMatch:
        """
        :param worded: The closest ready-made question, if any.
        :param similarity: How similar it is.
        :return: The match, or the reply that nothing matched closely enough.
        """
        if worded is None or similarity < self.minimum_similarity:
            return QuestionMatch(
                matched=False, similarity=similarity, reply=UNMATCHED_QUESTION_REPLY
            )
        return QuestionMatch(
            matched=True,
            similarity=similarity,
            preset=worded.preset.description(worded.verbalization),
        )

    def comparison(
        self, asked: SpokenText, worded: WordedPreset
    ) -> Tuple[float, float]:
        """
        Pair the best score with the negated count of words the wording adds beyond the
        question, so of two equally covered wordings the more specific one wins.

        :param asked: The question.
        :param worded: A ready-made question.
        :return: The score and the specificity.
        """
        said = asked.words()
        best = (0.0, -FULL_SIMILARITY)
        for wording in worded.wordings():
            words = SpokenText(wording).words()
            score = max(
                self.coverage(asked.spellings(), words),
                self.coverage(SpokenText(wording).spellings(), said),
            )
            pair = (score, -float(len(words) - len(said)))
            if pair > best:
                best = pair
        return best

    def coverage(self, spellings: FrozenSet[str], words: List[str]) -> float:
        """
        :param spellings: Every spelling the covering side offers.
        :param words: The words being covered.
        :return: How much of the words' weight the spellings cover, out of 100.
        """
        total = sum(self.weight(word) for word in set(words))
        if not total:
            return 0.0
        covered = {word for word in words if word in spellings}
        for first, second in zip(words, words[1:]):
            if first + second in spellings:
                covered |= {first, second}
        return FULL_SIMILARITY * sum(self.weight(word) for word in covered) / total
