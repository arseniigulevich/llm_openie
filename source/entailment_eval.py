import collections
import os
import threading
import numpy as np
from sentence_transformers import CrossEncoder
from scipy.special import softmax
from typing import Dict, Optional
from source.converter import combine_extraction_texts

DEFAULT_MODEL_NAME = 'cross-encoder/nli-deberta-v3-large'
_MODEL_CACHE: Dict[str, CrossEncoder] = {}
# The evaluator scores several systems in parallel threads. A Hugging Face fast tokenizer
# must not be used from two threads at once ("RuntimeError: Already borrowed"), and the
# long-pair path calls it directly, so all inference goes through this lock.
_PREDICT_LOCK = threading.Lock()
# Premise block size for pairs longer than the model input. NLI models trained on sentence
# pairs degrade on long premises (Laban et al. 2022): on MINE, precision fell to 0.25-0.40 with
# ~480-token premise windows but stayed at 0.82-0.97 with ~96-token blocks (about two to three
# sentences, or 8-10 triples), close to fully sentence-level scoring (2026-09-30).
# Every premise longer than one block goes through the block path, not only pairs over the
# model input: otherwise a premise of 100-500 tokens is read whole while a longer one is read
# in blocks, and a system's score depends on how much text it writes. On WiRe57 documents,
# the prototype's joined triples (220-490 tokens) were read whole and its combined recall
# fell to 0.007-0.234 on four of five documents, against 0.41-0.73 in blocks (2026-10-06).
PREMISE_BLOCK_TOKENS = 96


def _get_default_batch_size() -> int:
	"""Return entailment batch size from env with a safe default."""
	raw = os.environ.get('OPENIE_ENTAILMENT_BATCH_SIZE', '128')
	try:
		value = int(raw)
		return value if value > 0 else 128
	except (TypeError, ValueError):
		return 128


def get_cross_encoder(model_name: Optional[str] = None) -> CrossEncoder:
	"""Return a cached CrossEncoder instance, loading it on first use."""
	chosen_name = model_name or os.environ.get('OPENIE_ENTAILMENT_MODEL', DEFAULT_MODEL_NAME)
	if chosen_name not in _MODEL_CACHE:
		_MODEL_CACHE[chosen_name] = CrossEncoder(chosen_name)
	return _MODEL_CACHE[chosen_name]


def load_cross_encoder(model_name: Optional[str] = None) -> CrossEncoder:
	"""Load the cross-encoder onto the current CUDA device and update sota_model.

	Call this after selecting the target GPU (via CUDA_VISIBLE_DEVICES) and before
	running entailment evaluation.  The loaded model is stored in the module-level
	``sota_model`` variable so existing callers continue to work unchanged.
	"""
	global sota_model
	model = get_cross_encoder(model_name)
	sota_model = model
	print("Loaded cross-encoder for entailment evaluation.")
	return model


def unload_cross_encoder(model_name: Optional[str] = None) -> None:
	"""Move the cross-encoder off GPU and free both VRAM and system RAM.

	Call this immediately after all entailment work is done for a dataset so the
	VRAM is released for other users before the next stage begins.
	"""
	import gc
	global sota_model
	chosen_name = model_name or os.environ.get('OPENIE_ENTAILMENT_MODEL', DEFAULT_MODEL_NAME)
	model = _MODEL_CACHE.pop(chosen_name, None)
	sota_model = None
	if model is None:
		return
	try:
		import torch
		if hasattr(model, 'model'):
			# Move weights off GPU first to release VRAM.
			model.model.cpu()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			if hasattr(torch.cuda, 'ipc_collect'):
				torch.cuda.ipc_collect()
	except Exception:
		pass
	# Explicitly drop the last reference so the GC can reclaim system RAM now
	# rather than at some later indeterminate point.
	del model
	gc.collect()
	print(f"Unloaded cross-encoder '{chosen_name}' from GPU and RAM.")


# Intentionally NOT pre-loaded: call load_cross_encoder() after GPU selection.
sota_model = None


def read_extraction_file(filename, is_prediction, predictedsent2goldsent=None):
	sent2extraction = {}
	with open(filename, 'r') as f:
		for line in f:
			splitLine = line.strip().split('\t')
			if predictedsent2goldsent:
				if splitLine[0] in predictedsent2goldsent:
					sentence = predictedsent2goldsent[splitLine[0]]
				else:
					sentence = splitLine[0]
			else:
				sentence = splitLine[0]
			predicate = ''
			arg1 = ''
			arg2s = ['']
			if len(splitLine) > 1+is_prediction:
				predicate = splitLine[1+is_prediction]
			if len(splitLine) > 2+is_prediction:
				arg1 = splitLine[2+is_prediction]
			if len(splitLine) > 3+is_prediction:
				if is_prediction:
					arg2s = splitLine[4:7]
				else:
					arg2s = splitLine[3:]
			extraction = '{} {} {}'.format(arg1, predicate, ' '.join(arg2s))
			if sentence not in sent2extraction:
				sent2extraction[sentence] = []
			sent2extraction[sentence].append(extraction)
	return sent2extraction




label2prob = {'contradiction': 0, 'neutral': 2, 'entail': 1}
label_mapping = ['contradiction', 'entailment', 'neutral']


def _pair_limit(model: CrossEncoder) -> int:
	"""Longest (premise, hypothesis) pair the cross-encoder reads, in tokens."""
	limit = getattr(model, 'max_length', None) or getattr(model.tokenizer, 'model_max_length', 512)
	return int(min(limit, 512)) if limit and limit < 10**7 else 512


def _split_units(text: str) -> list:
	"""Sentences of a text (the triples of a joined extraction text end in '. ')."""
	import nltk
	try:
		units = nltk.sent_tokenize(text)
	except LookupError:
		nltk.download('punkt_tab', quiet=True)
		units = nltk.sent_tokenize(text)
	return [u for u in (x.strip() for x in units) if u]


def _token_pieces(tokenizer, text: str, max_tokens: int) -> list:
	"""Split a text into consecutive pieces of at most max_tokens tokens."""
	ids = tokenizer(text, add_special_tokens=False).input_ids
	return [tokenizer.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens)] or [text]


def _premise_pieces(tokenizer, premise: str, budget: int, cache: dict) -> tuple:
	"""Premise sentences (split further if longer than budget) and their token counts."""
	key = (premise, budget)
	if key not in cache:
		sents = cache.get(premise)
		if sents is None:
			sents = [(x, len(tokenizer(x, add_special_tokens=False).input_ids))
			         for x in (_split_units(premise) or [premise])]
			cache[premise] = sents
		pieces, lengths = [], []
		for sent, n in sents:
			if n > budget:
				for piece in _token_pieces(tokenizer, sent, budget):
					pieces.append(piece)
					lengths.append(len(tokenizer(piece, add_special_tokens=False).input_ids))
			else:
				pieces.append(sent)
				lengths.append(n)
		cache[key] = (pieces, lengths)
	return cache[key]


def _premise_windows(tokenizer, premise: str, budget: int, cache: Optional[dict] = None) -> list:
	"""Pack premise sentences greedily into windows of at most budget tokens.

	Consecutive windows share one sentence, so a fact that straddles a window
	boundary is still seen whole by one of them.
	"""
	cache = {} if cache is None else cache
	wkey = ('windows', premise, budget)
	if wkey in cache:
		return cache[wkey]
	pieces, lengths = _premise_pieces(tokenizer, premise, budget, cache)
	windows, start = [], 0
	while start < len(pieces):
		end, used = start, 0
		while end < len(pieces) and used + lengths[end] + 1 <= budget:
			used += lengths[end] + 1
			end += 1
		end = max(end, start + 1)
		windows.append(' '.join(pieces[start:end]))
		if end >= len(pieces):
			break
		start = end - 1 if end - 1 > start else end  # one sentence of overlap
	cache[wkey] = windows
	return windows


def _long_pair_units(tokenizer, premise: str, hypothesis: str, limit: int, cache: Optional[dict] = None):
	"""(premise window, hypothesis unit) pairs for one over-long example, plus unit ids."""
	cache = {} if cache is None else cache
	reserve = 4  # [CLS], [SEP], [SEP] and one spare
	units = []
	for unit in _split_units(hypothesis) or [hypothesis]:
		n = len(tokenizer(unit, add_special_tokens=False).input_ids)
		# keep at least half the window for the premise
		units.extend(_token_pieces(tokenizer, unit, limit // 2) if n > limit // 2 else [unit])
	pairs, unit_of_pair = [], []
	for u, unit in enumerate(units):
		budget = min(PREMISE_BLOCK_TOKENS, limit - reserve - len(tokenizer(unit, add_special_tokens=False).input_ids))
		for window in _premise_windows(tokenizer, premise, budget, cache):
			pairs.append((window, unit))
			unit_of_pair.append(u)
	return pairs, unit_of_pair, len(units)


def predict_entailment_probs(model: CrossEncoder, examples, batch_size: Optional[int] = None):
	"""Run batched CrossEncoder inference and return entailment probabilities.

	A pair whose premise fits one block (PREMISE_BLOCK_TOKENS tokens) and that fits the
	model's input (512 tokens for DeBERTa-v3) is scored as one pair. Any other pair is
	not truncated. Its hypothesis is split into sentences (for a joined extraction text,
	one triple each) and its premise into overlapping blocks of sentences of at most
	PREMISE_BLOCK_TOKENS tokens. Each unit
	scores the highest entailment probability over the premise windows, and the
	pair scores the mean over its units: the share of the hypothesis that some part
	of the premise entails.
	"""
	if not examples:
		return np.array([])
	with _PREDICT_LOCK:
		return _predict_entailment_probs_locked(model, examples, batch_size)


def _predict_entailment_probs_locked(model: CrossEncoder, examples, batch_size: Optional[int] = None):
	bs = batch_size or _get_default_batch_size()
	tokenizer = model.tokenizer
	limit = _pair_limit(model)

	short_idx, long_idx = [], []
	premise_len = {}
	for i, (premise, hypothesis) in enumerate(examples):
		n = len(tokenizer(premise, hypothesis, truncation=False).input_ids)
		if premise not in premise_len:
			premise_len[premise] = len(tokenizer(premise, add_special_tokens=False).input_ids)
		(short_idx if n <= limit and premise_len[premise] <= PREMISE_BLOCK_TOKENS else long_idx).append(i)

	def _entail(pairs):
		logits = model.predict(pairs, batch_size=bs, show_progress_bar=False, convert_to_numpy=True)
		return softmax(logits, axis=1)[:, label2prob['entail']]

	out = np.zeros(len(examples), dtype=float)
	if short_idx:
		out[short_idx] = _entail([examples[i] for i in short_idx])
	if long_idx:
		all_pairs, owner, cache = [], [], {}
		for i in long_idx:
			pairs, unit_of_pair, n_units = _long_pair_units(tokenizer, examples[i][0], examples[i][1], limit, cache)
			all_pairs.extend(pairs)
			owner.extend((i, u, n_units) for u in unit_of_pair)
		probs = _entail(all_pairs)
		best = collections.defaultdict(dict)
		for (i, u, _), prob in zip(owner, probs):
			best[i][u] = max(best[i].get(u, 0.0), float(prob))
		for i in long_idx:
			out[i] = float(np.mean(list(best[i].values()))) if best[i] else 0.0
	return out


def calculate_f1(p, r):
	return 2 * (float(p) * r) / (p + r)


def combine_tuples(tuple_list):
	return combine_extraction_texts(tuple_list)


def calculate_combined_tuple_tuple_score(sent2gold, sent2predictions, prediction_name, information_dict, model, sent_level_out_file):
	precision_examples = []
	id2sent = {}
	example_id = 0
	for sentence in sent2predictions:
		if sentence in sent2gold:
			combined_gold = combine_tuples(sent2gold[sentence])
		else:
			continue
		for predicted_tuple in sent2predictions[sentence]:
			if not len(predicted_tuple):
				continue
			precision_examples.append((combined_gold, predicted_tuple))
			id2sent[example_id] = sentence
			example_id += 1

	if len(precision_examples) == 0:
		entail_probs = np.array([])
	else:
		entail_probs = predict_entailment_probs(model, precision_examples)

	sent2precision_entailment = {}
	for i in range(len(entail_probs)):
		sentence = id2sent[i]
		if sentence not in sent2precision_entailment:
			sent2precision_entailment[sentence] = []
		sent2precision_entailment[sentence].append(float(entail_probs[i]))

	recall_examples = []
	id2sent = {}
	example_id = 0
	for sentence in sent2gold:
		if sentence in sent2predictions:
			combined_predictions = combine_tuples(sent2predictions[sentence])
		else:
			#combined_predictions = 'NA.'
			continue
		if not len(combined_predictions):
			continue
		for gold_tuple in sent2gold[sentence]:
			recall_examples.append((combined_predictions, gold_tuple))
			id2sent[example_id] = sentence
			example_id += 1

	if len(recall_examples) == 0:
		entail_probs = np.array([])
	else:
		try:
			entail_probs = predict_entailment_probs(model, recall_examples)
		except TypeError:
			entail_probs = np.array([0.0])

	sent2recall_entailment = {}
	for i in range(len(entail_probs)):
		sentence = id2sent[i]
		if sentence not in sent2recall_entailment:
			sent2recall_entailment[sentence] = []
		sent2recall_entailment[sentence].append(float(entail_probs[i]))

	print("Finished combined tuple to tuple")

	recalls = {}
	precisions = {}
	f1s = {}

	for sent in sent2recall_entailment.keys():
		if sent not in information_dict:
			information_dict[sent] = {}
		information_dict[sent]['combined_tuple_r'] = np.average(sent2recall_entailment[sent])
		information_dict[sent]['combined_tuple_individual_prediction_r'] = sent2recall_entailment[sent]
		recalls[sent] = np.average(sent2recall_entailment[sent])
	for sent in sent2precision_entailment.keys():
		if sent not in information_dict:
			information_dict[sent] = {}
		information_dict[sent]['combined_tuple_p'] = np.average(sent2precision_entailment[sent])
		information_dict[sent]['combined_tuple_individual_prediction_p'] = sent2precision_entailment[sent]
		precisions[sent] = np.average(sent2precision_entailment[sent])
	for sent in precisions.keys():
		if sent in recalls:
			f1s[sent] = calculate_f1(precisions[sent], recalls[sent])
			information_dict[sent]['combined_tuple_f1'] = calculate_f1(precisions[sent], recalls[sent])
			information_dict[sent]['combined_tuple_individual_prediction_f1'] = [calculate_f1(p, r) for p, r in zip(information_dict[sent]['combined_tuple_individual_prediction_p'], information_dict[sent]['combined_tuple_individual_prediction_r'])]
		else:
			f1s[sent] = calculate_f1(precisions[sent], 0)
			recalls[sent] = 0
			information_dict[sent]['combined_tuple_r'] = 0
			information_dict[sent]['combined_tuple_individual_prediction_r'] = [0]
			information_dict[sent]['combined_tuple_f1'] = calculate_f1(precisions[sent], 0)
			information_dict[sent]['combined_tuple_individual_prediction_f1'] = [calculate_f1(p, 0) for p in information_dict[sent]['combined_tuple_individual_prediction_p']]
	for sent in recalls.keys():
		if sent in precisions:
			if sent not in f1s:
				f1s[sent] = calculate_f1(precisions[sent], recalls[sent])
				information_dict[sent]['combined_tuple_f1'] = calculate_f1(precisions[sent], recalls[sent])
				information_dict[sent]['combined_tuple_individual_prediction_f1'] = [calculate_f1(p, r) for p, r in zip(information_dict[sent]['combined_tuple_individual_prediction_p'], information_dict[sent]['combined_tuple_individual_prediction_r'])]
		else:
			if sent not in f1s:
				f1s[sent] = calculate_f1(0, recalls[sent])
				information_dict[sent]['combined_tuple_f1'] = calculate_f1(0, recalls[sent])
				information_dict[sent]['combined_tuple_individual_prediction_f1'] = [calculate_f1(0, r) for r in zip(information_dict[sent]['combined_tuple_individual_prediction_r'])]
			precisions[sent] = 0.0
			information_dict[sent]['combined_tuple_p'] = 0.0
			information_dict[sent]['combined_tuple_individual_prediction_p'] = [0.0]

	#p, r, f1 = np.average(list(precisions.values())), np.average(list(recalls.values())), np.average(list(f1s.values()))
	p, r = np.average(list(precisions.values())), np.average(list(recalls.values()))
	f1 = calculate_f1(p, r)

	with open(sent_level_out_file, 'a') as f:
		f.write('{}\t{}\t{}\t{}\n'.format(prediction_name, p, r, f1))

	return p, r, f1


def calculate_sentence_tuple_score(sent2gold, sent2predictions, prediction_name, information_dict, model, sent_level_out_file):
	precision_examples = []
	id2sent = {}
	example_id = 0
	for sentence in sent2predictions:
		if sentence in sent2gold:
			combined_predictions = combine_tuples(sent2predictions[sentence])
			precision_examples.append((sentence, combined_predictions))
			id2sent[example_id] = sentence
			example_id += 1

	if len(precision_examples) == 0:
		entail_probs = np.array([])
	else:
		try:
			entail_probs = predict_entailment_probs(model, precision_examples)
		except TypeError:
			entail_probs = np.array([0.0])

	sent2precision_entailment = {}
	for i in range(len(entail_probs)):
		sentence = id2sent[i]
		if sentence not in sent2precision_entailment:
			sent2precision_entailment[sentence] = []
		sent2precision_entailment[sentence].append(float(entail_probs[i]))

	recall_examples = []
	id2sent = {}
	example_id = 0
	for sentence in sent2gold:
		if sentence in sent2predictions:
			combined_predictions = combine_tuples(sent2predictions[sentence])
			recall_examples.append((combined_predictions, sentence))
			id2sent[example_id] = sentence
			example_id += 1
		else:
			recall_examples.append(('N/A', sentence))
			id2sent[example_id] = sentence
			example_id += 1
	if len(recall_examples) == 0:
		entail_probs = np.array([])
	else:
		try:
			entail_probs = predict_entailment_probs(model, recall_examples)
		except TypeError:
			entail_probs = np.array([0.0])

	sent2recall_entailment = {}
	for i in range(len(entail_probs)):
		sentence = id2sent[i]
		if sentence not in sent2recall_entailment:
			sent2recall_entailment[sentence] = []
		sent2recall_entailment[sentence].append(float(entail_probs[i]))

	print("Finished sentence to tuple")

	recalls = {}
	precisions = {}
	f1s = {}

	for sent in sent2recall_entailment.keys():
		recalls[sent] = np.average(sent2recall_entailment[sent])
		if sent not in information_dict:
			information_dict[sent] = {}
		information_dict[sent]['sentence_r'] = np.average(sent2recall_entailment[sent])
		information_dict[sent]['sentence_individual_prediction_r'] = sent2recall_entailment[sent]
	for sent in sent2precision_entailment.keys():
		precisions[sent] = np.average(sent2precision_entailment[sent])
		if sent not in information_dict:
			information_dict[sent] = {}
		information_dict[sent]['sentence_p'] = np.average(sent2precision_entailment[sent])
		information_dict[sent]['sentence_individual_prediction_p'] = sent2precision_entailment[sent]
	for sent in precisions.keys():
		if sent in recalls:
			f1s[sent] = calculate_f1(precisions[sent], recalls[sent])
			information_dict[sent]['sentence_f1'] = calculate_f1(precisions[sent], recalls[sent])
			information_dict[sent]['sentence_individual_prediction_f1'] = [calculate_f1(p, r) for p, r in zip(information_dict[sent]['sentence_individual_prediction_p'], information_dict[sent]['sentence_individual_prediction_r'])]
		else:
			f1s[sent] = calculate_f1(precisions[sent], 0)
			recalls[sent] = 0
			information_dict[sent]['sentence_r'] = 0
			information_dict[sent]['sentence_individual_prediction_r'] = [0]
			information_dict[sent]['sentence_f1'] = calculate_f1(precisions[sent], 0)
			information_dict[sent]['sentence_individual_prediction_f1'] = [calculate_f1(p, 0) for p in information_dict[sent]['sentence_individual_prediction_p']]
	for sent in recalls.keys():
		if sent in precisions:
			if sent not in f1s:
				f1s[sent] = calculate_f1(precisions[sent], recalls[sent])
				information_dict[sent]['sentence_f1'] = calculate_f1(precisions[sent], recalls[sent])
				information_dict[sent]['sentence_individual_prediction_f1'] = [calculate_f1(p, r) for p, r in zip(information_dict[sent]['sentence_individual_prediction_p'], information_dict[sent]['sentence_individual_prediction_r'])]
		else:
			if sent not in f1s:
				f1s[sent] = calculate_f1(0, recalls[sent])
				information_dict[sent]['sentence_f1'] = calculate_f1(0, recalls[sent])
				information_dict[sent]['sentence_individual_prediction_f1'] = [calculate_f1(0, r) for r in information_dict[sent]['sentence_individual_prediction_r']]
			precisions[sent] = 0.0
			information_dict[sent]['sentence_p'] = 0.0
			information_dict[sent]['sentence_individual_prediction_p'] = [0.0]

	#p, r, f1 = np.average(list(precisions.values())), np.average(list(recalls.values())), np.average(list(f1s.values()))
	p, r = np.average(list(precisions.values())), np.average(list(recalls.values()))
	f1 = calculate_f1(p, r)

	with open(sent_level_out_file, 'a') as f:
		f.write('{}\t{}\t{}\t{}\n'.format(prediction_name, p, r, f1))

	return p, r, f1


def calculate_tuple_tuple_score(sent2gold, sent2predictions, prediction_name, information_dict, model, sent_level_out_file):
	precision_examples = []
	id2sent = {}
	sent2id = {}
	example_id = 0

	unmatched_precision_predictions = {}
	for sent in sent2gold:
		unmatched_precision_predictions[sent] = []
		for gold_tuple in sent2gold[sent]:
			unmatched_precision_predictions[sent].append(gold_tuple)

	for sentence in sent2predictions:
		for predicted_tuple in sent2predictions[sentence]:
			if not len(predicted_tuple):
				continue
			if sentence in sent2gold:
				for gold_tuple in sent2gold[sentence]:
					precision_examples.append((gold_tuple, predicted_tuple))
					id2sent[example_id] = sentence
					if sentence not in sent2id:
						sent2id[sentence] = []
					sent2id[sentence].append(example_id)
					example_id += 1

	if len(precision_examples) == 0:
		entail_probs = np.array([])
	else:
		entail_probs = predict_entailment_probs(model, precision_examples)
		
	sent2precision_entailment = {}
	matched_precision_predictions = {}
	for sent in sent2id:
		scores = {}
		predictions = set()
		for example_id in sent2id[sent]:
			gold, prediction = precision_examples[example_id]
			predictions.add(prediction)
			if gold not in scores:
				scores[gold] = {}
			scores[gold][prediction] = float(entail_probs[example_id])

		selected_rows = []
		selected_cols = []
		num_precision_matches = min(len(scores), len(predictions))
		precision_numerator = 0
		for t in range(num_precision_matches):
			matched_row = -1
			matched_col = -1
			matched_precision = -np.inf # initialised to <0 so that it updates whenever precision is 0 as well
			for gold in scores:
				if gold in selected_rows:
					continue
				for prediction in predictions:
					if prediction in selected_cols:
						continue
					if scores[gold][prediction] > matched_precision:
						matched_precision = scores[gold][prediction]
						matched_row = gold
						matched_col = prediction

			selected_rows.append(matched_row)
			selected_cols.append(matched_col)
			if sent not in matched_precision_predictions:
				matched_precision_predictions[sent] = []
			#matched_precision_predictions[sent][matched_row] = matched_col
			matched_precision_predictions[sent].append((matched_row, matched_col))
			unmatched_precision_predictions[sent].remove(matched_row)
			#precision_numerator += scores[matched_row][matched_col]
			if sent not in sent2precision_entailment:
				sent2precision_entailment[sent] = []
			sent2precision_entailment[sent].append(scores[matched_row][matched_col])

	recall_examples = []
	id2sent = {}
	sent2id = {}
	example_id = 0
	for sentence in sent2gold:
		for gold_tuple in sent2gold[sentence]:
			if sentence in sent2predictions:
				for predicted_tuple in sent2predictions[sentence]:
					if not len(predicted_tuple):
						continue
					recall_examples.append((predicted_tuple, gold_tuple))
					id2sent[example_id] = sentence
					if sentence not in sent2id:
						sent2id[sentence] = []
					sent2id[sentence].append(example_id)
					example_id += 1
			else:
				recall_examples.append(('', gold_tuple))
				id2sent[example_id] = sentence
				if sentence not in sent2id:
					sent2id[sentence] = []
				sent2id[sentence].append(example_id)
				example_id += 1

	if len(recall_examples) == 0:
		entail_probs = np.array([])
	else:
		entail_probs = predict_entailment_probs(model, recall_examples)
		
	sent2recall_entailment = {}
	matched_recall_predictions = {}
	for sent in sent2id:
		scores = {}
		if sent not in matched_recall_predictions:
			matched_recall_predictions[sent] = []
		for example_id in sent2id[sent]:
			prediction, gold = recall_examples[example_id]
			if gold not in scores:
				scores[gold] = []
			scores[gold].append((prediction, float(entail_probs[example_id])))
		if sent not in sent2recall_entailment:
			sent2recall_entailment[sent] = []
		for gold in scores:
			prediction, max_score = max(scores[gold], key=lambda x: x[1])
			sent2recall_entailment[sent].append(max_score)
			if isinstance(prediction, str) and prediction.strip():
				matched_recall_predictions[sent].append(prediction)
			else:
				matched_recall_predictions[sent].append('')

	print("Finished tuple to tuple")

	recalls = {}
	precisions = {}
	f1s = {}

	for sent in unmatched_precision_predictions:
		if sent not in information_dict:
			information_dict[sent] = {}
		information_dict[sent]['unmatched_precision_predictions'] = unmatched_precision_predictions[sent]

	for sent in sent2recall_entailment.keys():
		recalls[sent] = np.average(sent2recall_entailment[sent])
		if sent not in information_dict:
			information_dict[sent] = {}
		information_dict[sent]['tuple_r'] = np.average(sent2recall_entailment[sent])
		information_dict[sent]['tuple_individual_prediction_r'] = sent2recall_entailment[sent]
		information_dict[sent]['matched_recall_predictions'] = matched_recall_predictions[sent]
	for sent in sent2precision_entailment.keys():
		precisions[sent] = np.average(sent2precision_entailment[sent])
		if sent not in information_dict:
			information_dict[sent] = {}
		information_dict[sent]['tuple_p'] = np.average(sent2precision_entailment[sent])
		information_dict[sent]['tuple_individual_prediction_p'] = sent2precision_entailment[sent]
		information_dict[sent]['matched_precision_predictions'] = matched_precision_predictions[sent]
	for sent in precisions.keys():
		if sent in recalls:
			f1s[sent] = calculate_f1(precisions[sent], recalls[sent])
			information_dict[sent]['tuple_f1'] = calculate_f1(precisions[sent], recalls[sent])
			information_dict[sent]['tuple_individual_prediction_f1'] = [calculate_f1(p, r) for p, r in zip(information_dict[sent]['tuple_individual_prediction_p'], information_dict[sent]['tuple_individual_prediction_r'])]
		else:
			f1s[sent] = calculate_f1(precisions[sent], 0)
			recalls[sent] = 0
			information_dict[sent]['tuple_r'] = 0
			information_dict[sent]['tuple_individual_prediction_r'] = [0]
			information_dict[sent]['tuple_f1'] = calculate_f1(precisions[sent], 0)
			information_dict[sent]['tuple_individual_prediction_f1'] = [calculate_f1(p, 0) for p in information_dict[sent]['tuple_individual_prediction_p']]
	for sent in recalls.keys():
		if sent in precisions:
			if sent not in f1s:
				f1s[sent] = calculate_f1(precisions[sent], recalls[sent])
				information_dict[sent]['tuple_f1'] = calculate_f1(precisions[sent], recalls[sent])
				information_dict[sent]['tuple_individual_prediction_f1'] = [calculate_f1(p, r) for p, r in zip(information_dict[sent]['tuple_individual_prediction_p'], information_dict[sent]['tuple_individual_prediction_r'])]
		else:
			if sent not in f1s:
				f1s[sent] = calculate_f1(0, recalls[sent])
				information_dict[sent]['tuple_f1'] = calculate_f1(0, recalls[sent])
				information_dict[sent]['tuple_individual_prediction_f1'] = [calculate_f1(0, r) for r in information_dict[sent]['tuple_individual_prediction_r']]
			precisions[sent] = 0.0
			information_dict[sent]['tuple_p'] = 0.0
			information_dict[sent]['tuple_individual_prediction_p'] = [0.0]

	#p, r, f1 = np.average(list(precisions.values())), np.average(list(recalls.values())), np.average(list(f1s.values()))
	p, r = np.average(list(precisions.values())), np.average(list(recalls.values()))
	f1 = calculate_f1(p, r)

	with open(sent_level_out_file, 'a') as f:
		f.write('{}\t{}\t{}\t{}\n'.format(prediction_name, p, r, f1))

	return p, r, f1
	

