from __future__ import annotations

import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass(slots=True)
class EnvironmentConfig:
    hf_token: str | None
    hf_home: str | None
    hf_local_files_only: bool
    torch_device: str
    torch_dtype: str


@dataclass(slots=True)
class ModelConfig:
    base_model_id: str
    scope_release: str
    scope_width: str
    layer_selection: str | list[int]


@dataclass(slots=True)
class TokenizationConfig:
    seq_len: int
    add_special_tokens: bool


@dataclass(slots=True)
class AppConfig:
    """Model and tokenization settings required by feature extraction."""

    env: EnvironmentConfig
    model: ModelConfig
    tokenization: TokenizationConfig


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_model_config(
    config_path: str | Path,
    env_path: str | Path | None = None,
) -> AppConfig:
    """Load model settings without importing the adapter from CLI code."""

    if env_path is not None:
        load_dotenv(env_path, override=False)
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"model configuration not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("model configuration root must be a mapping")

    model_raw = _mapping(raw.get("model"), "model")
    token_raw = _mapping(raw.get("tokenization"), "tokenization")
    base_model_id = str(model_raw.get("base_model_id", "")).strip()
    if not base_model_id:
        raise ValueError("model.base_model_id is required")
    seq_len = int(token_raw.get("seq_len", 1024))
    if seq_len <= 0:
        raise ValueError("tokenization.seq_len must be positive")

    environment = EnvironmentConfig(
        hf_token=_optional_env("HF_TOKEN"),
        hf_home=_optional_env("HF_HOME"),
        hf_local_files_only=_bool_env("HF_LOCAL_FILES_ONLY", False),
        torch_device=os.getenv("TORCH_DEVICE", "auto"),
        torch_dtype=os.getenv("TORCH_DTYPE", "bfloat16"),
    )
    if environment.hf_home:
        os.environ.setdefault("HF_HOME", environment.hf_home)
    if environment.hf_local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    layer_selection: str | list[int] = model_raw.get("layer_selection", "all")
    if isinstance(layer_selection, list):
        layer_selection = [int(value) for value in layer_selection]
    elif not isinstance(layer_selection, str):
        raise ValueError("model.layer_selection must be 'all' or a list of layer indices")

    return AppConfig(
        env=environment,
        model=ModelConfig(
            base_model_id=base_model_id,
            scope_release=str(model_raw.get("scope_release", "")),
            scope_width=str(model_raw.get("scope_width", "")),
            layer_selection=layer_selection,
        ),
        tokenization=TokenizationConfig(
            seq_len=seq_len,
            add_special_tokens=bool(token_raw.get("add_special_tokens", False)),
        ),
    )


@dataclass(slots=True)
class WindowBatch:
    input_ids: list[int]
    tokens: list[str]
    text: str
    window_start: int
    window_end: int
    token_sentence_ids: list[int]
    sentence_spans: list[dict[str, Any]]
    clause_spans: list[dict[str, Any]]


class GemmaModelAdapter:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.env.torch_device)
        self.dtype = self._resolve_dtype(config.env.torch_dtype)
        self.model_source = self._resolve_model_source()
        self._loaded_with_device_map = False
        self._spacy_nlp: Any | None = None

        self.model_config = AutoConfig.from_pretrained(
            self.model_source,
            token=config.env.hf_token,
            local_files_only=config.env.hf_local_files_only,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            token=config.env.hf_token,
            local_files_only=config.env.hf_local_files_only,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_load_kwargs: dict[str, Any] = {
            "token": config.env.hf_token,
            "local_files_only": config.env.hf_local_files_only,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            model_load_kwargs["device_map"] = {"": "cuda:0"}
            self._loaded_with_device_map = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_source,
            **model_load_kwargs,
        )
        if not self._loaded_with_device_map:
            self._place_model()
        self.model.eval()

    def tokenize_document(self, text: str) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=self.config.tokenization.add_special_tokens,
            return_attention_mask=False,
            return_tensors=None,
        )
        return list(encoded["input_ids"])

    def make_windows(
        self,
        token_ids: list[int],
        metadata_mode: str = "heuristic",
        window_len: int | None = None,
    ) -> list[WindowBatch]:
        seq_len = window_len or self.config.tokenization.seq_len
        windows: list[WindowBatch] = []
        for start in range(0, len(token_ids), seq_len):
            chunk = token_ids[start : start + seq_len]
            if not chunk:
                continue
            text = self.tokenizer.decode(chunk, skip_special_tokens=False)
            tokens = self.tokenizer.convert_ids_to_tokens(chunk)
            token_sentence_ids, sentence_spans, clause_spans = self._window_text_metadata(
                token_ids=chunk,
                text=text,
                metadata_mode=metadata_mode,
            )
            windows.append(
                WindowBatch(
                    input_ids=chunk,
                    tokens=tokens,
                    text=text,
                    window_start=start,
                    window_end=start + len(chunk),
                    token_sentence_ids=token_sentence_ids,
                    sentence_spans=sentence_spans,
                    clause_spans=clause_spans,
                )
            )
        return windows

    def max_context_window_tokens(self) -> int:
        max_positions = getattr(self.model_config, "max_position_embeddings", None)
        if isinstance(max_positions, int) and max_positions > 0:
            return max_positions
        return self.config.tokenization.seq_len

    def forward_outputs(
        self,
        input_ids: list[int],
        require_grad: bool,
        layer_output_addition: tuple[int, torch.Tensor, list[int] | str | None] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        tensor = torch.tensor([input_ids], device=self.device)
        context = nullcontext() if require_grad else torch.no_grad()
        hook_handle = None
        if layer_output_addition is not None:
            layer_idx, delta, positions = layer_output_addition
            layer_module = self._layer_module(int(layer_idx))
            hook_handle = layer_module.register_forward_hook(
                self._make_layer_output_addition_hook(
                    delta=delta,
                    positions=positions,
                )
            )
        try:
            with context:
                outputs = self.model(
                    input_ids=tensor,
                    output_hidden_states=True,
                    use_cache=False,
                )
        finally:
            if hook_handle is not None:
                hook_handle.remove()
        return outputs, self.describe_model()

    def _layer_module(self, layer_idx: int) -> Any:
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            raise RuntimeError("Could not resolve decoder layers on the current model.")
        return layers[layer_idx]

    def _make_layer_output_addition_hook(
        self,
        delta: torch.Tensor,
        positions: list[int] | str | None,
    ):
        delta_tensor = delta.to(self.device, dtype=self.dtype).reshape(1, 1, -1)

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden_states = output[0] if isinstance(output, tuple) else output
            if hidden_states.ndim != 3:
                return output
            modified = hidden_states.clone()
            token_positions = self._resolve_steering_positions(hidden_states.shape[1], positions)
            if token_positions:
                modified[:, token_positions, :] = modified[:, token_positions, :] + delta_tensor
            if isinstance(output, tuple):
                return (modified, *output[1:])
            return modified

        return hook

    @staticmethod
    def _resolve_steering_positions(
        sequence_length: int,
        positions: list[int] | str | None,
    ) -> list[int]:
        if sequence_length <= 0:
            return []
        if positions is None or positions == "all":
            return list(range(sequence_length))
        if positions == "last":
            return [sequence_length - 1]
        resolved: list[int] = []
        for position in positions:
            index = int(position)
            if 0 <= index < sequence_length and index not in resolved:
                resolved.append(index)
        return resolved

    def describe_model(self) -> dict[str, Any]:
        cfg = self.model_config
        return {
            "model_id": self.config.model.base_model_id,
            "hidden_size": getattr(cfg, "hidden_size", None),
            "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
            "num_attention_heads": getattr(cfg, "num_attention_heads", None),
            "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
            "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
            "sliding_window": getattr(cfg, "sliding_window", None),
            "vocab_size": getattr(cfg, "vocab_size", None),
            "torch_device": str(self.device),
            "torch_dtype": str(self.dtype).replace("torch.", ""),
        }

    def prefetch(self) -> dict[str, Any]:
        return self.describe_model()

    def _place_model(self) -> None:
        try:
            self.model.to(self.device)
        except RuntimeError as exc:
            if self.device == "mps":
                self.device = "cpu"
                self.model.to(self.device)
                return
            raise exc

    def _resolve_model_source(self) -> str:
        if not self.config.env.hf_local_files_only:
            return self.config.model.base_model_id

        return snapshot_download(
            repo_id=self.config.model.base_model_id,
            token=self.config.env.hf_token,
            local_files_only=True,
        )

    @staticmethod
    def _resolve_device(requested: str) -> str:
        lowered = requested.lower()
        if lowered != "auto":
            return lowered
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _resolve_dtype(requested: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        lowered = requested.lower()
        if lowered not in mapping:
            raise ValueError(f"Unsupported TORCH_DTYPE: {requested}")
        return mapping[lowered]

    def _window_text_metadata(
        self,
        token_ids: list[int],
        text: str,
        metadata_mode: str = "heuristic",
    ) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
        if metadata_mode == "spacy":
            return self._window_text_metadata_spacy(token_ids=token_ids, text=text)
        if metadata_mode != "heuristic":
            raise ValueError(f"Unsupported metadata mode: {metadata_mode}")

        return self._window_text_metadata_heuristic(token_ids=token_ids, text=text)

    def _window_text_metadata_heuristic(
        self,
        token_ids: list[int],
        text: str,
    ) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
            return_tensors=None,
        )
        offsets = list(encoded.get("offset_mapping") or [])
        encoded_ids = list(encoded.get("input_ids") or [])

        if len(encoded_ids) != len(token_ids):
            return self._fallback_single_sentence_metadata(len(token_ids))

        sentence_char_spans = self._split_text_spans(text, boundary_pattern=r"(?<=[.!?])\s+")
        sentence_spans, token_sentence_ids = self._map_char_spans_to_tokens(
            char_spans=sentence_char_spans,
            offsets=offsets,
            span_type="sentence",
        )
        if not sentence_spans:
            return self._fallback_single_sentence_metadata(len(token_ids))

        clause_spans = self._build_clause_spans(text=text, offsets=offsets, sentence_spans=sentence_spans)
        return token_sentence_ids, sentence_spans, clause_spans

    def _window_text_metadata_spacy(
        self,
        token_ids: list[int],
        text: str,
    ) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
            return_tensors=None,
        )
        offsets = list(encoded.get("offset_mapping") or [])
        encoded_ids = list(encoded.get("input_ids") or [])

        if len(encoded_ids) != len(token_ids):
            return self._fallback_single_sentence_metadata(len(token_ids))

        nlp = self._get_spacy_nlp()
        doc = nlp(text)
        sentence_char_spans = [
            (sent.start_char, sent.end_char)
            for sent in doc.sents
            if sent.text.strip()
        ]
        if not sentence_char_spans:
            return self._fallback_single_sentence_metadata(len(token_ids))

        sentence_spans, token_sentence_ids = self._map_char_spans_to_tokens(
            char_spans=sentence_char_spans,
            offsets=offsets,
            span_type="sentence",
        )
        if not sentence_spans:
            return self._fallback_single_sentence_metadata(len(token_ids))

        clause_spans = self._build_clause_spans_spacy(
            doc=doc,
            offsets=offsets,
            sentence_spans=sentence_spans,
        )
        if not clause_spans:
            clause_spans = self._build_clause_spans(
                text=text,
                offsets=offsets,
                sentence_spans=sentence_spans,
            )
        return token_sentence_ids, sentence_spans, clause_spans

    def _build_clause_spans(
        self,
        text: str,
        offsets: list[tuple[int, int]],
        sentence_spans: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        clauses: list[dict[str, Any]] = []
        clause_id = 0
        for sentence in sentence_spans:
            sent_start_char = int(sentence["start_char"])
            sent_end_char = int(sentence["end_char"])
            spans = self._split_text_spans(
                text[sent_start_char:sent_end_char],
                boundary_pattern=r"[,;:]+|(?:\s[—-]\s)",
                keep_separator_out=True,
                base_offset=sent_start_char,
            )
            if not spans:
                spans = [(sent_start_char, sent_end_char)]
            mapped_spans, _ = self._map_char_spans_to_tokens(
                char_spans=spans,
                offsets=offsets,
                span_type="clause",
                starting_span_id=clause_id,
                sentence_id=int(sentence["sentence_id"]),
            )
            clause_id += len(mapped_spans)
            clauses.extend(mapped_spans)
        return clauses

    def _build_clause_spans_spacy(
        self,
        doc: Any,
        offsets: list[tuple[int, int]],
        sentence_spans: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        clauses: list[dict[str, Any]] = []
        clause_id = 0
        sentence_by_id = {
            int(sentence["sentence_id"]): sentence
            for sentence in sentence_spans
        }

        for sentence in doc.sents:
            if not sentence.text.strip():
                continue
            sentence_span = self._find_sentence_span(sentence_by_id, sentence.start_char, sentence.end_char)
            if sentence_span is None:
                continue
            sentence_id = int(sentence_span["sentence_id"])
            clause_char_spans = self._spacy_clause_char_spans(sentence)
            mapped_spans, _ = self._map_char_spans_to_tokens(
                char_spans=clause_char_spans,
                offsets=offsets,
                span_type="clause",
                starting_span_id=clause_id,
                sentence_id=sentence_id,
            )
            if not mapped_spans:
                mapped_spans, _ = self._map_char_spans_to_tokens(
                    char_spans=[(sentence.start_char, sentence.end_char)],
                    offsets=offsets,
                    span_type="clause",
                    starting_span_id=clause_id,
                    sentence_id=sentence_id,
                )
            clause_id += len(mapped_spans)
            clauses.extend(mapped_spans)
        return clauses

    def _map_char_spans_to_tokens(
        self,
        char_spans: list[tuple[int, int]],
        offsets: list[tuple[int, int]],
        span_type: str,
        starting_span_id: int = 0,
        sentence_id: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[int]]:
        spans: list[dict[str, Any]] = []
        token_span_ids = [-1 for _ in offsets]

        next_span_id = starting_span_id
        for start_char, end_char in char_spans:
            overlapping = [
                token_idx
                for token_idx, (token_start, token_end) in enumerate(offsets)
                if token_end > start_char and token_start < end_char
            ]
            if not overlapping:
                continue
            start_token_index = overlapping[0]
            end_token_index = overlapping[-1]
            for token_idx in overlapping:
                token_span_ids[token_idx] = next_span_id
            span_record = {
                f"{span_type}_id": next_span_id,
                "start_char": start_char,
                "end_char": end_char,
                "start_token_index": start_token_index,
                "end_token_index": end_token_index,
            }
            if span_type == "sentence":
                span_record["sentence_id"] = next_span_id
            if sentence_id is not None:
                span_record["sentence_id"] = sentence_id
            spans.append(span_record)
            next_span_id += 1

        if span_type == "sentence":
            for token_idx, span_id in enumerate(token_span_ids):
                if span_id < 0:
                    token_span_ids[token_idx] = spans[-1]["sentence_id"] if spans else 0
        return spans, token_span_ids

    @staticmethod
    def _split_text_spans(
        text: str,
        boundary_pattern: str,
        keep_separator_out: bool = True,
        base_offset: int = 0,
    ) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = 0
        for match in re.finditer(boundary_pattern, text):
            start = cursor
            end = match.start() if keep_separator_out else match.end()
            if text[start:end].strip():
                trimmed_start = start + len(text[start:end]) - len(text[start:end].lstrip())
                trimmed_end = end - len(text[start:end]) + len(text[start:end].rstrip())
                spans.append((base_offset + trimmed_start, base_offset + trimmed_end))
            cursor = match.end()
        tail = text[cursor:]
        if tail.strip():
            trimmed_start = cursor + len(tail) - len(tail.lstrip())
            trimmed_end = cursor + len(tail.rstrip())
            spans.append((base_offset + trimmed_start, base_offset + trimmed_end))
        return spans

    @staticmethod
    def _fallback_single_sentence_metadata(
        token_count: int,
    ) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
        if token_count <= 0:
            return [], [], []
        sentence_span = {
            "sentence_id": 0,
            "start_char": 0,
            "end_char": 0,
            "start_token_index": 0,
            "end_token_index": token_count - 1,
        }
        clause_span = {
            "clause_id": 0,
            "sentence_id": 0,
            "start_char": 0,
            "end_char": 0,
            "start_token_index": 0,
            "end_token_index": token_count - 1,
        }
        return [0 for _ in range(token_count)], [sentence_span], [clause_span]

    def _get_spacy_nlp(self) -> Any:
        if self._spacy_nlp is not None:
            return self._spacy_nlp
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError(
                "Transcript discovery requires spaCy. Install project deps and "
                "then run `python -m spacy download en_core_web_sm`."
            ) from exc
        try:
            self._spacy_nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "Transcript discovery requires the spaCy model `en_core_web_sm`. "
                "Install it with `python -m spacy download en_core_web_sm`."
            ) from exc
        return self._spacy_nlp

    @staticmethod
    def _find_sentence_span(
        sentence_by_id: dict[int, dict[str, Any]],
        start_char: int,
        end_char: int,
    ) -> dict[str, Any] | None:
        for sentence in sentence_by_id.values():
            if start_char >= int(sentence["start_char"]) and end_char <= int(sentence["end_char"]):
                return sentence
        return None

    @staticmethod
    def _spacy_clause_char_spans(sentence: Any) -> list[tuple[int, int]]:
        clause_head_deps = {
            "ROOT",
            "advcl",
            "acl",
            "ccomp",
            "xcomp",
            "relcl",
            "parataxis",
            "conj",
            "csubj",
            "csubjpass",
        }
        sentence_start = sentence.start_char
        sentence_end = sentence.end_char
        spans: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        for token in sentence:
            if token.pos_ not in {"VERB", "AUX"}:
                continue
            if token.dep_ not in clause_head_deps:
                continue
            subtree_tokens = list(token.subtree)
            if not subtree_tokens:
                continue
            start_char = max(sentence_start, subtree_tokens[0].idx)
            end_char = min(sentence_end, subtree_tokens[-1].idx + len(subtree_tokens[-1]))
            if end_char <= start_char:
                continue
            span = (start_char, end_char)
            if span in seen:
                continue
            seen.add(span)
            spans.append(span)

        if not spans:
            return [(sentence_start, sentence_end)]
        return sorted(spans)


class TextModelAdapter(GemmaModelAdapter):
    """Load text-only checkpoints, including multimodal-packaged Gemma 3 models."""

    def __init__(self, config: AppConfig) -> None:
        model_id = str(config.model.base_model_id)
        if not model_id.startswith("google/gemma-3-"):
            super().__init__(config)
            return

        self.config = config
        self.device = self._resolve_device(config.env.torch_device)
        self.dtype = self._resolve_dtype(config.env.torch_dtype)
        self.model_source = self._resolve_model_source()
        self._loaded_with_device_map = False
        self._spacy_nlp = None

        outer_config = AutoConfig.from_pretrained(
            self.model_source,
            token=config.env.hf_token,
            local_files_only=config.env.hf_local_files_only,
        )
        self.model_config = getattr(outer_config, "text_config", outer_config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            token=config.env.hf_token,
            local_files_only=config.env.hf_local_files_only,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "token": config.env.hf_token,
            "local_files_only": config.env.hf_local_files_only,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            load_kwargs["device_map"] = {"": "cuda:0"}
            self._loaded_with_device_map = True

        if hasattr(outer_config, "vision_config"):
            from transformers import Gemma3ForConditionalGeneration

            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_source,
                **load_kwargs,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_source,
                **load_kwargs,
            )
        if not self._loaded_with_device_map:
            self._place_model()
        self.model.eval()

    def _layer_module(self, layer_idx: int) -> Any:
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            multimodal_model = getattr(self.model, "model", None)
            language_model = getattr(multimodal_model, "language_model", None)
            layers = getattr(language_model, "layers", None)
        if layers is None:
            raise RuntimeError("could not resolve decoder layers on the current model")
        return layers[layer_idx]
