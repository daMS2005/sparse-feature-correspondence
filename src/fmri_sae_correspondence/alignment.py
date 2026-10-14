from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TranscriptPaths:
    stimulus_id: str
    transcript_txt: Path
    words_tsv: Path
    tr_aligned_tsv: Path
    metadata_json: Path


@dataclass(frozen=True, slots=True)
class WordTiming:
    word: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class TrBin:
    tr_index: int
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True, slots=True)
class TokenWordGroup:
    text: str
    token_indices: tuple[int, ...]


def resolve_transcript_paths(transcript_root: Path, stimulus_id: str) -> TranscriptPaths:
    base = Path(transcript_root) / stimulus_id
    paths = TranscriptPaths(
        stimulus_id=stimulus_id,
        transcript_txt=base / f"{stimulus_id}_transcript.txt",
        words_tsv=base / f"{stimulus_id}_words.tsv",
        tr_aligned_tsv=base / f"{stimulus_id}_tr_aligned.tsv",
        metadata_json=base / "metadata.json",
    )
    for path in (
        paths.transcript_txt,
        paths.words_tsv,
        paths.tr_aligned_tsv,
        paths.metadata_json,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def load_word_rows(path: Path) -> list[WordTiming]:
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: list[WordTiming] = []
        for row in reader:
            word = str(row["word"])
            if not normalize_text(word):
                continue
            rows.append(
                WordTiming(
                    word=word,
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                )
            )
        return rows


def load_tr_bins(path: Path) -> list[TrBin]:
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [
            TrBin(
                tr_index=int(row["tr_index"]),
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
                text=str(row["text"]),
            )
            for row in reader
        ]


def group_tokens_with_model_tokenizer(
    model_id: str,
    tokens: list[str],
    transcript_text: str,
    tokenizer: Any,
) -> list[TokenWordGroup]:
    encoded = tokenizer(transcript_text, add_special_tokens=False, return_attention_mask=False)
    encoded_ids = encoded["input_ids"]
    encoded_tokens = list(tokenizer.convert_ids_to_tokens(encoded_ids))
    if encoded_tokens != tokens:
        for index, (expected, observed) in enumerate(zip(tokens, encoded_tokens, strict=False)):
            if expected != observed:
                raise ValueError(
                    f"tokenizer mismatch for {model_id} at token {index}: "
                    f"artifact={expected!r} tokenizer={observed!r}"
                )
        if len(encoded_tokens) != len(tokens):
            raise ValueError(
                f"tokenizer mismatch for {model_id}: artifact has {len(tokens)} tokens, "
                f"tokenizer produced {len(encoded_tokens)}"
            )
    return group_tokens_into_words(
        encoded_tokens,
        word_start_marker=infer_word_start_marker(encoded_tokens, model_id=model_id),
    )


def infer_word_start_marker(tokens: list[str], model_id: str = "") -> str | None:
    normalized_model_id = model_id.lower()
    if "gemma" in normalized_model_id:
        return "▁"
    if "llama" in normalized_model_id:
        return "Ġ"
    if any(token.startswith("Ġ") for token in tokens):
        return "Ġ"
    if any(token.startswith("▁") for token in tokens):
        return "▁"
    return None


def _is_tokenizer_whitespace_token(token: str) -> bool:
    return token in {"Ċ", "<0x0A>", "\n", "\r", "\t"}


def group_tokens_into_words(
    tokens: list[str],
    word_start_marker: str | None = None,
) -> list[TokenWordGroup]:
    groups: list[TokenWordGroup] = []
    current_text: list[str] = []
    current_indices: list[int] = []
    start_marker = (
        word_start_marker
        if word_start_marker is not None
        else infer_word_start_marker(tokens)
    )
    for index, token in enumerate(tokens):
        if _is_tokenizer_whitespace_token(token):
            if current_text:
                groups.append(
                    TokenWordGroup(
                        text="".join(current_text),
                        token_indices=tuple(current_indices),
                    )
                )
                current_text = []
                current_indices = []
            continue
        is_word_start = bool(start_marker) and token.startswith(start_marker)
        if is_word_start and current_text:
            groups.append(
                TokenWordGroup(
                    text="".join(current_text),
                    token_indices=tuple(current_indices),
                )
            )
            current_text = []
            current_indices = []
        piece = token[len(start_marker) :] if is_word_start and start_marker else token
        if not piece and not normalize_text(token):
            continue
        current_text.append(piece)
        current_indices.append(index)
    if current_text:
        groups.append(
            TokenWordGroup(
                text="".join(current_text),
                token_indices=tuple(current_indices),
            )
        )
    return groups


def validate_word_alignment(
    token_word_groups: list[TokenWordGroup],
    word_rows: list[WordTiming],
    transcript_text: str,
) -> None:
    if len(token_word_groups) != len(word_rows):
        raise ValueError(
            f"token-derived word count {len(token_word_groups)} does not match "
            f"timed words {len(word_rows)}"
        )
    mismatches: list[str] = []
    for index, (token_word, timed_word) in enumerate(zip(token_word_groups, word_rows, strict=False)):
        if normalize_text(token_word.text) != normalize_text(timed_word.word):
            mismatches.append(f"{index}:{token_word.text!r}!={timed_word.word!r}")
            if len(mismatches) >= 5:
                break
    if mismatches:
        raise ValueError("token-to-word alignment mismatches: " + ", ".join(mismatches))
    reconstructed = " ".join(word.word for word in word_rows).strip()
    if normalize_text(reconstructed) != normalize_text(transcript_text):
        raise ValueError("timed words do not reconstruct the transcript text")


def build_token_to_word_index(
    token_word_groups: list[TokenWordGroup],
) -> dict[int, int]:
    token_to_word: dict[int, int] = {}
    for word_index, group in enumerate(token_word_groups):
        for token_index in group.token_indices:
            token_to_word[token_index] = word_index
    return token_to_word


def assign_words_to_trs(
    word_rows: list[WordTiming],
    tr_bins: list[TrBin],
    stimulus_onset_s: float,
) -> list[int]:
    if not tr_bins:
        raise ValueError("at least one TR bin is required")
    assignments: list[int] = []
    for word in word_rows:
        midpoint = stimulus_onset_s + ((word.start_s + word.end_s) / 2.0)
        assigned = None
        for tr_index, tr_bin in enumerate(tr_bins):
            if tr_bin.start_s <= midpoint < tr_bin.end_s:
                assigned = tr_index
                break
        if assigned is None and math.isclose(midpoint, tr_bins[-1].end_s):
            assigned = len(tr_bins) - 1
        if assigned is None:
            raise ValueError(
                f"could not assign word {word.word!r} at midpoint={midpoint} to a TR bin"
            )
        assignments.append(int(assigned))
    return assignments


def validate_tr_alignment(
    word_rows: list[WordTiming],
    word_to_tr_index: list[int],
    tr_bins: list[TrBin],
    stimulus_onset_s: float,
) -> None:
    by_tr: dict[int, list[str]] = defaultdict(list)
    for word, tr_index in zip(word_rows, word_to_tr_index, strict=False):
        by_tr[int(tr_index)].append(word.word)
    mismatches: list[str] = []
    for tr_index, tr_bin in enumerate(tr_bins):
        expected = normalize_text(tr_bin.text)
        observed = normalize_text(" ".join(by_tr.get(tr_index, [])))
        if expected != observed:
            mismatches.append(f"TR {tr_index}: {observed!r}!={expected!r}")
            if len(mismatches) >= 5:
                break
    if mismatches:
        raise ValueError("word-to-TR alignment mismatches: " + ", ".join(mismatches))
    if stimulus_onset_s > 0 and tr_bins and tr_bins[0].start_s != 0.0:
        raise ValueError("TR bins are expected to start at 0.0 seconds")


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())
