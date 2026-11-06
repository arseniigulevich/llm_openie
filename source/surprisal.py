"""
Surprisal-based OpenIE Evaluation Module

This module evaluates OpenIE extractions by measuring how "surprising" the original 
sentence is when conditioned on the extracted tuples. The idea is that good extractions 
should capture the essential information, making the original sentence less surprising.

Lower surprisal = tuples capture more information from the original sentence.
Higher reduction ratio = better extractions.

Long inputs: nothing is truncated as long as BOS + context + target fits the
model's context window (model.config.max_position_embeddings, capped by the
tokenizer limit). Qwen/Qwen2.5-1.5B (32k tokens) covers the longest inputs in the
thesis (a MINE article with 11k tokens of triples). Only if a pair exceeds the
window is the context cut from the left, keeping the tokens nearest the target;
target tokens are never dropped.

Every target token is scored, including the first: each sequence starts with the
model's BOS token (or EOS where there is none), so the unconditional and the
conditional surprisal average over the same tokens.

Copy-rate correction (2026-10-03): both directions reward triples that copy the
source sentence (experiments_ranking_v2/copyrate_correction). Each extraction set's
S-r and S-p are multiplied by its atomicity weight

    A = mean over triples t of 1 - max(0, copy(t) - C0),    C0 = 0.5

where copy(t) is the share of the best-matching source sentence that t contains
(multiset token overlap / sentence length; documents are split into sentences).
S-f1 is the harmonic mean of the corrected S-r and S-p, i.e. A * raw S-f1. The
uncorrected values are kept in the details as ss_*_raw.
"""

import re
from collections import Counter

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import threading

COPY_THRESHOLD = 0.5
_TOKEN_RE = re.compile(r"\w+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")

# Global model cache to avoid reloading
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_RUN_LOCKS: Dict[str, threading.Lock] = {}


def _get_model_run_lock(model_name: str) -> threading.Lock:
    """Get/create a per-model lock to avoid concurrent CUDA inference issues."""
    with _MODEL_CACHE_LOCK:
        if model_name not in _MODEL_RUN_LOCKS:
            _MODEL_RUN_LOCKS[model_name] = threading.Lock()
        return _MODEL_RUN_LOCKS[model_name]


def unload_surprisal_model(model_name: str = "gpt2") -> None:
    """Move the surprisal model off GPU and free both VRAM and system RAM.

    Call this immediately after all SS work is done for a dataset so the VRAM
    is released for other users before the next stage begins.
    """
    import gc as _gc
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.pop(model_name, None)
        _MODEL_RUN_LOCKS.pop(model_name, None)
    if cached is None:
        return
    model, _tokenizer, _device = cached
    try:
        # Move weights off GPU first to release VRAM.
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    # Explicitly drop the last references so the GC can reclaim system RAM now.
    del model, _tokenizer, cached
    _gc.collect()
    print(f"Unloaded surprisal model '{model_name}' from GPU and RAM.")


def _load_model(model_name: str = "gpt2"):
    """Load a causal language model and tokenizer (cached)."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    with _MODEL_CACHE_LOCK:
        if model_name in _MODEL_CACHE:
            return _MODEL_CACHE[model_name]

        print(f"Loading surprisal model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Force full materialization of weights to avoid meta tensor CUDA errors.
        # float32 on purpose: surprisal scores are ratios of small differences between
        # mean log-losses, and bf16 makes those depend on batch composition (padding).
        model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=False, torch_dtype=torch.float32)

        # Ensure tied output/input embeddings for Causal LM heads (e.g., GPT-2 family).
        # This avoids using a randomly initialized lm_head when some checkpoints omit it.
        try:
            model.tie_weights()
        except Exception:
            pass

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            try:
                model = model.cuda()
                print("Using CUDA for surprisal computation")
            except NotImplementedError:
                # Fallback for edge cases where tensors are still on meta device.
                print("Warning: CUDA transfer failed due meta tensor; falling back to CPU for SS.")
                device = "cpu"
                model = model.cpu()

        _MODEL_CACHE[model_name] = (model, tokenizer, device)
        return model, tokenizer, device


def _tuple_to_text(triple: List[str]) -> str:
    """Convert a relational tuple (subject, predicate, and optional objects) to natural text."""
    if len(triple) >= 2:
        return " ".join(str(t).strip() for t in triple)
    return " ".join([str(t).strip() for t in triple])


def _tuples_to_text(tuples: List[List[str]]) -> str:
    """Convert multiple tuples to a combined text representation."""
    tuple_texts = [_tuple_to_text(t) for t in tuples if t]
    return " . ".join(tuple_texts) + " ." if tuple_texts else ""


def _clamp_01(value: float) -> float:
    """Clamp numeric value to [0, 1]."""
    return max(0.0, min(1.0, value))


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def copy_share(triple: List[str], sentence_counters: List[Counter]) -> Optional[float]:
    """Share of the best-matching source sentence the triple contains (multiset overlap)."""
    t = Counter(_tokens(" ".join(str(e) for e in triple)))
    shares = [sum((t & s).values()) / sum(s.values()) for s in sentence_counters if sum(s.values())]
    return max(shares) if shares else None


def atomicity_weight(text: str, tuples: List[List[str]],
                     threshold: float = COPY_THRESHOLD) -> Optional[float]:
    """A = mean over triples of 1 - max(0, copy - threshold); None if undefined
    (no triples, or a source text without tokens)."""
    sents = [Counter(_tokens(s)) for s in _SENT_SPLIT.split(text)] or [Counter(_tokens(text))]
    shares = [c for c in (copy_share(t, sents) for t in tuples if t) if c is not None]
    if not shares:
        return None
    return float(np.mean([1.0 - max(0.0, c - threshold) for c in shares]))


def copy_correct(recall: float, precision: float, weight: Optional[float]) -> Tuple[float, float, float]:
    """Corrected (S-r, S-p, S-f1). An undefined weight leaves the scores unchanged."""
    a = 1.0 if weight is None else weight
    r, p = recall * a, precision * a
    return r, p, (2 * p * r / (p + r) if (p + r) > 0 else 0.0)


def _context_limit(model, tokenizer) -> int:
    """Longest sequence the model accepts (positions), capped by the tokenizer."""
    limit = int(getattr(model.config, "max_position_embeddings", 1024) or 1024)
    tok_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_limit, int) and 0 < tok_limit < 10**7:
        limit = min(limit, tok_limit)
    return limit


def _mean_target_nll(
    model,
    tokenizer,
    contexts: List[Optional[str]],
    targets: List[str],
    device: str,
    token_budget: int = 32768,
    max_batch: int = 16,
    logit_chunk: int = 2048,
) -> List[float]:
    """Mean surprisal (nats per token) of each target given its context.

    Sequence = [BOS] + context + target, tokenised separately and concatenated, so
    the boundary between context and target is exact. A context of None scores the
    target unconditionally (given only BOS). Logits are computed only for target
    positions, in chunks, so long sequences do not materialise a full
    (length x vocabulary) matrix.
    """
    if len(contexts) != len(targets):
        raise ValueError("contexts and targets must have the same length")
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else bos
    limit = _context_limit(model, tokenizer)

    seqs = []  # (input_ids, index of first target token)
    for ctx, tgt in zip(contexts, targets):
        ctx_ids = tokenizer(ctx, add_special_tokens=False).input_ids if ctx else []
        tgt_ids = tokenizer((" " + tgt) if ctx else tgt, add_special_tokens=False).input_ids
        if 1 + len(tgt_ids) > limit:
            tgt_ids = tgt_ids[: limit - 1]
            print(f"Warning: surprisal target longer than {limit} tokens; truncated.")
        room = limit - 1 - len(tgt_ids)
        if len(ctx_ids) > room:
            ctx_ids = ctx_ids[len(ctx_ids) - room:]  # keep the context nearest the target
        seqs.append(([bos] + ctx_ids + tgt_ids, 1 + len(ctx_ids)))

    results: List[Optional[float]] = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    base = model.base_model
    head = model.get_output_embeddings()
    i = 0
    while i < len(order):
        # grow the batch while it fits the token budget
        j = i + 1
        while (j < len(order) and j - i < max_batch
               and len(seqs[order[j]][0]) * (j - i + 1) <= token_budget):
            j += 1
        batch = order[i:j]
        width = max(len(seqs[k][0]) for k in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, k in enumerate(batch):
            seq = seqs[k][0]
            ids[row, :len(seq)] = torch.tensor(seq)
            mask[row, :len(seq)] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad():
            hidden = base(input_ids=ids, attention_mask=mask).last_hidden_state
            for row, k in enumerate(batch):
                seq, start = seqs[k]
                if len(seq) - start <= 0:
                    results[k] = 0.0
                    continue
                # position p predicts token p + 1
                pos = torch.arange(start - 1, len(seq) - 1, device=device)
                labels = ids[row, pos + 1]
                total = 0.0
                for c in range(0, len(pos), logit_chunk):
                    logits = head(hidden[row, pos[c:c + logit_chunk]]).float()
                    total += torch.nn.functional.cross_entropy(
                        logits, labels[c:c + logit_chunk], reduction="sum").item()
                results[k] = total / len(pos)
        del hidden
        i = j
    return results


def calculate_surprisal_score(
    data_to_evaluate: Dict[str, List[List[str]]],
    output_all_scores: bool = False,
    model_name: str = "gpt2",
    batch_size: int = 16,
    copy_correction: bool = True,
) -> Tuple[
    float,
    Dict,
    Dict[str, float],
    float,
    float,
    Dict[str, float],
    Dict[str, float],
]:
    
    """
    Calculate surprisal-based evaluation scores for OpenIE extractions.
    
    For each sentence, compute:
    1. Compute unconditional surprisal of the sentence
    2. Convert tuples to text
    3. SS-recall via conditional surprisal: P(sentence | tuples)
    4. SS-precision via conditional surprisal: P(tuples | sentence)
    5. Copy-rate correction: SS-recall and SS-precision times the atomicity weight A
    6. SS-F1 as harmonic mean of the corrected SS-precision and SS-recall

    A good extraction should reduce surprisal significantly (higher reduction ratio).

    Args:
        data_to_evaluate: Dict mapping sentences to list of triples
        output_all_scores: Whether to return detailed per-sentence scores
        model_name: Name of the causal LM model to use (default: gpt2)
        copy_correction: Apply the copy-rate correction (default True); False returns
            the raw scores
    
    Returns:
        Tuple of (
            average_ss_recall,
            details_dict,
            text2ss_recall,
            average_ss_precision,
            average_ss_f1,
            text2ss_precision,
            text2ss_f1,
        )
    """
    
    model, tokenizer, device = _load_model(model_name)
    run_lock = _get_model_run_lock(model_name)

    with run_lock:
        details = {}
        text2score = {}
        text2precision = {}
        text2f1 = {}
        recall_ratios = []
        precision_ratios = []
        f1_scores = []

        sentences = list(data_to_evaluate.keys())

        valid_sentences = []
        valid_tuple_texts = []

        for sentence in tqdm(sentences, desc="Preparing SS inputs"):
            tuples = data_to_evaluate[sentence]

            if not tuples:
                text2score[sentence] = 0.0
                text2precision[sentence] = 0.0
                text2f1[sentence] = 0.0
                if output_all_scores:
                    details[sentence] = {
                        "num_tuples": 0,
                        "ss_recall": 0.0,
                        "ss_precision": 0.0,
                        "ss_f1": 0.0,
                        "ss_recall_raw": 0.0,
                        "ss_precision_raw": 0.0,
                        "ss_f1_raw": 0.0,
                        "atomicity": None,
                        "unconditional_surprisal": None,
                        "conditional_surprisal": None,
                    }
                continue

            tuple_text = _tuples_to_text(tuples)
            if not tuple_text.strip():
                text2score[sentence] = 0.0
                text2precision[sentence] = 0.0
                text2f1[sentence] = 0.0
                if output_all_scores:
                    details[sentence] = {
                        "num_tuples": len(tuples),
                        "ss_recall": 0.0,
                        "ss_precision": 0.0,
                        "ss_f1": 0.0,
                        "ss_recall_raw": 0.0,
                        "ss_precision_raw": 0.0,
                        "ss_f1_raw": 0.0,
                        "atomicity": None,
                        "unconditional_surprisal": None,
                        "conditional_surprisal": None,
                    }
                continue

            valid_sentences.append(sentence)
            valid_tuple_texts.append(tuple_text)

        if valid_sentences:
            sentence_unconditional_vals = _mean_target_nll(
                model, tokenizer, [None] * len(valid_sentences), valid_sentences, device, max_batch=batch_size
            )
            sentence_conditional_vals = _mean_target_nll(
                model, tokenizer, valid_tuple_texts, valid_sentences, device, max_batch=batch_size
            )
            tuple_unconditional_vals = _mean_target_nll(
                model, tokenizer, [None] * len(valid_tuple_texts), valid_tuple_texts, device, max_batch=batch_size
            )
            tuple_conditional_vals = _mean_target_nll(
                model, tokenizer, valid_sentences, valid_tuple_texts, device, max_batch=batch_size
            )

            for i, sentence in enumerate(valid_sentences):
                tuple_text = valid_tuple_texts[i]
                tuples = data_to_evaluate[sentence]

                sentence_unconditional = sentence_unconditional_vals[i]
                sentence_conditional = sentence_conditional_vals[i]
                tuple_unconditional = tuple_unconditional_vals[i]
                tuple_conditional = tuple_conditional_vals[i]

                sentence_reduction = sentence_unconditional - sentence_conditional
                recall_ratio = sentence_reduction / sentence_unconditional if sentence_unconditional > 0 else 0.0

                tuple_reduction = tuple_unconditional - tuple_conditional
                precision_ratio = tuple_reduction / tuple_unconditional if tuple_unconditional > 0 else 0.0

                recall_raw = _clamp_01(recall_ratio)
                precision_raw = _clamp_01(precision_ratio)
                f1_raw = (2 * precision_raw * recall_raw / (precision_raw + recall_raw)
                          if (precision_raw + recall_raw) > 0 else 0.0)

                weight = atomicity_weight(sentence, tuples) if copy_correction else None
                recall_ratio, precision_ratio, ss_f1 = copy_correct(recall_raw, precision_raw, weight)

                text2score[sentence] = recall_ratio
                text2precision[sentence] = precision_ratio
                text2f1[sentence] = ss_f1
                recall_ratios.append(recall_ratio)
                precision_ratios.append(precision_ratio)
                f1_scores.append(ss_f1)

                if output_all_scores:
                    details[sentence] = {
                        "num_tuples": len(tuples),
                        "tuple_text": tuple_text,
                        "sentence_unconditional_surprisal": sentence_unconditional,
                        "sentence_conditional_surprisal": sentence_conditional,
                        "sentence_surprisal_reduction": sentence_reduction,
                        "tuple_unconditional_surprisal": tuple_unconditional,
                        "tuple_conditional_surprisal": tuple_conditional,
                        "tuple_surprisal_reduction": tuple_reduction,
                        "ss_recall": recall_ratio,
                        "ss_precision": precision_ratio,
                        "ss_f1": ss_f1,
                        "ss_recall_raw": recall_raw,
                        "ss_precision_raw": precision_raw,
                        "ss_f1_raw": f1_raw,
                        "atomicity": weight,
                    }

        # Compute aggregate score
        avg_reduction_ratio = float(np.mean(recall_ratios)) if recall_ratios else 0.0
        avg_precision_ratio = float(np.mean(precision_ratios)) if precision_ratios else 0.0
        avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

        return (
            avg_reduction_ratio,
            details,
            text2score,
            avg_precision_ratio,
            avg_f1,
            text2precision,
            text2f1,
        )
