from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Ambiguous terms: ML model/paper names that collide with common English words or
# named entities. Each entry maps the lowercased ambiguous token to its candidate
# senses. The first sense is the ML/paper sense (preferred in a scholarly context).
AMBIGUOUS_TERMS: Dict[str, List[str]] = {
    # Pre-existing
    "transformer": ["ML Transformer models", "Electrical power transformers"],
    "python": ["Python programming language", "Python snake"],
    "apple": ["Apple Inc.", "apple fruit"],
    "jaguar": ["Jaguar animal", "Jaguar car brand"],
    "java": ["Java programming language", "Java island/coffee"],
    "rust": ["Rust programming language", "rust corrosion"],
    "spark": ["Apache Spark", "electric spark"],
    "shell": ["Unix shell", "shell (physical)"],
    "bert": ["BERT language model", "person/entity named Bert"],
    "git": ["Git version control", "git as noun/other"],
    "linux": ["Linux operating system", "Linux distribution ecosystem"],
    "stream": ["data stream/computing", "natural stream"],
    "node": ["Node.js runtime", "graph/node concept"],
    "react": ["React framework", "react verb/chemistry"],
    # ML paper / model names that collide with people, products, or common words
    "colbert": ["ColBERT retrieval model", "Stephen Colbert (comedian)"],
    "rag": ["Retrieval-Augmented Generation", "rag / cloth"],
    "bart": ["BART (denoising seq2seq model)", "person named Bart"],
    "pegasus": ["PEGASUS summarization model", "Pegasus (mythical horse)"],
    "clip": ["CLIP (contrastive vision-language model)", "paper clip / video clip"],
    "adam": ["Adam optimizer", "person named Adam"],
    "whisper": ["Whisper (speech recognition model)", "whisper (speak quietly)"],
    "dalle": ["DALL-E (text-to-image)", "Salvador Dalí reference"],
    "gemini": ["Gemini (Google LLM)", "Gemini zodiac / spacecraft"],
    "llama": ["LLaMA (Meta LLM)", "llama animal"],
    "palm": ["PaLM (Google LLM)", "palm tree / body part"],
    "sparse": ["sparse retrieval (BM25/TF-IDF)", "sparse (adjective)"],
    "dense": ["dense retrieval / dual encoder", "dense (adjective)"],
    "attention": ["attention mechanism (neural networks)", "attention (cognitive)"],
    "vit": ["ViT (Vision Transformer)", "vit (other abbreviation)"],
    "gpt": ["GPT language model", "general-purpose transformer / other"],
    "yolo": ["YOLO object detector", "YOLO (slang)"],
    "t5": ["T5 text-to-text transformer", "T5 (other designation)"],
    "dpr": ["DPR (Dense Passage Retrieval)", "DPR (other acronym)"],
    "rlhf": ["RLHF (reinforcement learning from human feedback)", "RLHF (other)"],
    "squad": ["SQuAD reading-comprehension dataset", "squad (group of people)"],
    "beir": ["BEIR retrieval benchmark", "BEIR (other)"],
    "drqa": ["DrQA open-domain QA system", "DrQA (other)"],
    "instructgpt": ["InstructGPT (RLHF-tuned GPT)", "InstructGPT (other)"],
    "chain": ["chain-of-thought prompting", "chain (physical)"],
    "mamba": ["Mamba state-space model", "mamba (snake)"],
    "unet": ["U-Net segmentation architecture", "U-Net (other)"],
    "resnet": ["ResNet residual network", "ResNet (other)"],
    "faiss": ["FAISS similarity-search library", "FAISS (other)"],
    "ann": ["Approximate Nearest Neighbor search", "Artificial Neural Network (generic)"],
    "ner": ["Named Entity Recognition", "NER (other)"],
    "qa": ["Question Answering", "Quality Assurance"],
    "ocr": ["Optical Character Recognition", "OCR (other)"],
    "cnn": ["Convolutional Neural Network", "CNN news network"],
    "rnn": ["Recurrent Neural Network", "RNN (other)"],
    "lstm": ["LSTM recurrent architecture", "LSTM (other)"],
    "gan": ["Generative Adversarial Network", "gan (other)"],
    "factscore": ["FActScore atomic-fact scoring", "FActScore (other)"],
    "chatbot": ["chatbot (conversational AI)", "chatbot (other)"],
    "ir": ["Information Retrieval", "Infrared (IR)"],
    "nlp": ["Natural Language Processing", "Neuro-Linguistic Programming"],
    "cv": ["Computer Vision", "curriculum vitae"],
}

# Keywords that help disambiguate a detected sense. The resolver scores each sense
# by how many of these keywords appear in retrieved-chunk context OR — with the
# new query-rewriting function — in the literal query itself.
SENSE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "ML Transformer models": (
        "transformer",
        "attention",
        "llm",
        "nlp",
        "bert",
        "gpt",
        "encoder",
        "decoder",
        "self-attention",
        "multi-head",
    ),
    "Electrical power transformers": (
        "electrical",
        "power",
        "voltage",
        "substation",
        "thermal",
        "condition monitoring",
        "grid",
        "circuit",
    ),
    "ColBERT retrieval model": (
        "retrieval",
        "passage",
        "dense",
        "late interaction",
        "bert",
        "information retrieval",
        "ir",
        "ranking",
        "embedding",
    ),
    "Stephen Colbert (comedian)": ("comedian", "television", "daily show", "late show", "satire", "political"),
    "Retrieval-Augmented Generation": ("retrieval", "generation", "knowledge", "wikipedia", "seq2seq", "dpr", "bart"),
    "BART (denoising seq2seq model)": (
        "denoising",
        "seq2seq",
        "generation",
        "summarization",
        "encoder-decoder",
        "pretraining",
    ),
    "PEGASUS summarization model": ("summarization", "gap-sentence", "abstractive", "pretraining", "news"),
    "CLIP (contrastive vision-language model)": (
        "contrastive",
        "image",
        "vision",
        "zero-shot",
        "caption",
        "image-text",
    ),
    "Adam optimizer": ("optimizer", "gradient", "learning rate", "training", "adaptive", "stochastic"),
    "Whisper (speech recognition model)": ("speech", "audio", "recognition", "asr", "transcription", "openai"),
    "LLaMA (Meta LLM)": ("language model", "llm", "meta", "pretraining", "transformer", "instruction"),
    "attention mechanism (neural networks)": (
        "attention",
        "query",
        "key",
        "value",
        "transformer",
        "softmax",
        "self-attention",
    ),
    "ViT (Vision Transformer)": ("vision", "image", "patch", "transformer", "classification", "pretraining"),
    "GPT language model": ("generative", "language model", "transformer", "pretraining", "gpt", "openai"),
    "YOLO object detector": ("object detection", "bounding box", "real-time", "detector", "anchors"),
    "T5 text-to-text transformer": ("text-to-text", "transfer learning", "pretraining", "span corruption"),
    "DPR (Dense Passage Retrieval)": ("dense", "passage", "dual encoder", "retrieval", "bert", "open-domain"),
    "RLHF (reinforcement learning from human feedback)": (
        "reinforcement learning",
        "reward model",
        "ppo",
        "alignment",
        "preference",
    ),
    "SQuAD reading-comprehension dataset": (
        "reading comprehension",
        "wikipedia",
        "crowd",
        "span",
        "question answering",
    ),
    "BEIR retrieval benchmark": ("benchmark", "zero-shot", "retrieval", "bm25", "evaluation", "heterogeneous"),
    "DrQA open-domain QA system": ("wikipedia", "tf-idf", "open-domain", "question answering", "reader"),
    "InstructGPT (RLHF-tuned GPT)": ("instruction", "rlhf", "reward", "alignment", "openai", "human feedback"),
    "chain-of-thought prompting": ("prompting", "reasoning", "step-by-step", "intermediate", "few-shot"),
    "U-Net segmentation architecture": (
        "segmentation",
        "biomedical",
        "skip connection",
        "encoder-decoder",
        "convolutional",
    ),
    "ResNet residual network": ("residual", "skip connection", "deep network", "image", "classification"),
    "FAISS similarity-search library": ("similarity search", "vector index", "ann", "nearest neighbor", "embedding"),
    "Convolutional Neural Network": ("convolution", "cnn", "image", "feature map", "pooling"),
    "Information Retrieval": ("retrieval", "bm25", "tf-idf", "ranking", "index", "query"),
    "Natural Language Processing": ("nlp", "language", "tokenization", "parsing", "semantic"),
    "FActScore atomic-fact scoring": ("factuality", "atomic fact", "wikipedia", "verification", "long-form"),
    "Generative Adversarial Network": ("generator", "discriminator", "adversarial", "gan", "minimax"),
}

# Word sets that, when found in the query, strongly signal a scholarly / ML
# context. Used to bias the query-rewrite in favor of the ML sense.
ML_CONTEXT_SIGNALS: Tuple[str, ...] = (
    "paper",
    "model",
    "architecture",
    "network",
    "neural",
    "deep learning",
    "machine learning",
    "ml",
    "ai",
    "nlp",
    "retrieval",
    "encoder",
    "decoder",
    "transformer",
    "pretraining",
    "fine-tune",
    "fine-tuning",
    "benchmark",
    "dataset",
    "training",
    "inference",
    "embedding",
    "attention",
    "token",
    "language model",
    "llm",
    "vision",
    "segmentation",
    "summarization",
    "question answering",
    "qa",
    "reasoning",
    "chain of thought",
    "research",
)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _detect_term(query: str) -> str | None:
    q = _tokens(query)
    for t in AMBIGUOUS_TERMS.keys():
        if t in q or (t + "s") in q:
            return t
    return None


def _score_sense(sense: str, snippets: List[str]) -> float:
    keys = SENSE_KEYWORDS.get(sense, ())
    if not keys:
        return 0.0
    joined = " ".join(snippets).lower()
    return float(sum(1 for k in keys if k in joined))


def _query_has_ml_context(query: str) -> bool:
    q = (query or "").lower()
    return any(signal in q for signal in ML_CONTEXT_SIGNALS)


def expand_query_for_ml_sense(query: str, *, scholarly_default: bool = True) -> dict:
    """Pre-retrieval query rewrite.

    If the query contains an ambiguous term and is short or low-context,
    rewrite it to append ML-sense keywords so that dense retrieval surfaces
    the right paper/chunk. Returns a dict:
        {
            "expanded_query": str,   # rewritten query (or original)
            "term": str | None,
            "ml_sense": str | None,  # the ML sense that was boosted
            "rewritten": bool,
            "reason": str,
        }

    `scholarly_default=True` means: if we cannot tell from the query text,
    default to the ML sense (the first sense for each term). This is what we
    want inside a scholarly-RAG product.
    """
    if not query or not query.strip():
        return {
            "expanded_query": query or "",
            "term": None,
            "ml_sense": None,
            "rewritten": False,
            "reason": "empty-query",
        }

    term = _detect_term(query)
    if not term:
        return {
            "expanded_query": query,
            "term": None,
            "ml_sense": None,
            "rewritten": False,
            "reason": "no-ambiguous-term",
        }

    options = AMBIGUOUS_TERMS.get(term, [])
    if len(options) < 1:
        return {
            "expanded_query": query,
            "term": term,
            "ml_sense": None,
            "rewritten": False,
            "reason": "no-senses-configured",
        }

    # Score each sense using the query itself (not retrieved chunks — we haven't
    # retrieved yet). For a short query like "tell me about Colbert" the scores
    # will be zero across senses, which is exactly when we need to default.
    query_lower = query.lower()
    scores = [(opt, _score_sense(opt, [query_lower])) for opt in options]
    top_score = scores[0][1]
    second_score = scores[1][1] if len(scores) > 1 else 0.0

    ml_sense = options[0]  # ML sense is always first per AMBIGUOUS_TERMS contract
    word_count = len(query.split())
    has_ml_context = _query_has_ml_context(query)
    ml_sense_wins = scores[0][0] == ml_sense and top_score >= second_score

    # Decide whether to rewrite. We rewrite to BOOST the ML sense in retrieval
    # when any of the following holds:
    #   (a) Query has an explicit ML / scholarly signal ("paper", "model", ...).
    #   (b) Query is short (<= 7 words) — too little context for plain dense
    #       retrieval to disambiguate; we default to the ML sense in a
    #       scholarly-RAG product when `scholarly_default=True`.
    #   (c) ML-sense keyword evidence in the query itself already wins over
    #       the alternative senses.
    should_rewrite = False
    reason = "no-rewrite"
    if has_ml_context:
        should_rewrite = True
        reason = "ml-context-signal"
    elif word_count <= 7 and scholarly_default:
        should_rewrite = True
        reason = "short-ambiguous-default-ml"
    elif ml_sense_wins and scholarly_default:
        should_rewrite = True
        reason = "ml-sense-wins-in-query"
    elif scholarly_default and top_score == 0 and second_score == 0:
        should_rewrite = True
        reason = "no-query-evidence-default-ml"

    if not should_rewrite:
        return {
            "expanded_query": query,
            "term": term,
            "ml_sense": ml_sense,
            "rewritten": False,
            "reason": reason,
        }

    boost_keys = SENSE_KEYWORDS.get(ml_sense, ())
    # Limit boost to ~6 strong keywords so we don't drown the original query.
    boost = " ".join(list(boost_keys)[:6])
    # Preserve the original query (the embedding model handles redundancy fine).
    expanded = f"{query} — context: {ml_sense}. {boost}"
    return {
        "expanded_query": expanded,
        "term": term,
        "ml_sense": ml_sense,
        "rewritten": True,
        "reason": reason,
    }


def resolve_sense(query: str, top_chunks: List[Dict], chosen_sense: str | None = None) -> Dict:
    term = _detect_term(query)
    if not term:
        return {
            "is_ambiguous": False,
            "term": None,
            "options": [],
            "rationale": "No curated ambiguous term detected.",
            "recommended_option": None,
        }

    options = AMBIGUOUS_TERMS.get(term, [])
    snippets = [f"{c.get('title', '')} {c.get('snippet', '')}" for c in (top_chunks or [])[:8]]

    sense_scores = [(opt, _score_sense(opt, snippets)) for opt in options]
    sense_scores.sort(key=lambda x: x[1], reverse=True)
    evidence_top = sense_scores[0][0] if sense_scores else None
    evidence_top_score = sense_scores[0][1] if sense_scores else 0.0

    if chosen_sense and chosen_sense in options:
        return {
            "is_ambiguous": False,
            "term": term,
            "options": options,
            "rationale": f"User selected sense: {chosen_sense}.",
            "recommended_option": chosen_sense,
        }

    if len(options) < 2:
        return {
            "is_ambiguous": False,
            "term": term,
            "options": options,
            "rationale": "Only one sense configured.",
            "recommended_option": options[0] if options else None,
        }

    ml_sense = options[0]
    query_scores = [(opt, _score_sense(opt, [query.lower()])) for opt in options]
    query_scores.sort(key=lambda x: x[1], reverse=True)
    query_top = query_scores[0][0] if query_scores else None
    query_top_score = query_scores[0][1] if query_scores else 0.0

    # If the query itself clearly signals the scholarly / ML sense, do not bounce
    # the user into a clarification loop just because the evidence snippets are
    # sparse or the alternative sense is generic.
    if (
        query_top == ml_sense
        and (_query_has_ml_context(query) or query_top_score >= 2.0)
        and (evidence_top == ml_sense or evidence_top_score <= 0.0)
    ):
        return {
            "is_ambiguous": False,
            "term": term,
            "options": options,
            "rationale": f"Query text itself strongly favors the scholarly sense '{ml_sense}'.",
            "recommended_option": ml_sense,
            "sense_scores": [{"sense": s, "score": float(sc)} for s, sc in sense_scores],
        }

    nonzero = [sc for _, sc in sense_scores if sc > 0]
    # If two+ senses are present in evidence, ask clarification.
    if len(nonzero) >= 2:
        is_ambiguous = True
    else:
        # If evidence strongly supports one sense we can proceed, otherwise ask.
        top = sense_scores[0][1]
        second = sense_scores[1][1] if len(sense_scores) > 1 else 0.0
        is_ambiguous = (top - second) <= 1.0

    return {
        "is_ambiguous": bool(is_ambiguous),
        "term": term,
        "options": options,
        "rationale": f"Detected ambiguous term '{term}' with close sense evidence scores.",
        "recommended_option": sense_scores[0][0] if sense_scores else None,
        "sense_scores": [{"sense": s, "score": float(sc)} for s, sc in sense_scores],
    }


def filter_citations_by_sense(citations: List[Dict], sense: str | None) -> List[Dict]:
    if not sense:
        return citations
    keys = SENSE_KEYWORDS.get(sense, ())
    if not keys:
        return citations
    out = []
    for c in citations:
        hay = f"{c.get('title', '')} {c.get('snippet', '')}".lower()
        if any(k in hay for k in keys):
            out.append(c)
    return out or citations


def _word_hit(haystack_tokens: set[str], haystack_lower: str, phrase: str) -> bool:
    """Word/phrase boundary match. Avoids 'bert' matching inside 'Colbert'."""
    phrase = phrase.lower().strip()
    if not phrase:
        return False
    if " " in phrase or "-" in phrase:
        # Multi-word phrase: require the full phrase to appear verbatim.
        return phrase in haystack_lower
    return phrase in haystack_tokens


def is_offtopic_public_result(query: str, citation: dict) -> bool:
    """Public-search domain prior.

    Return True if a public-search citation appears to be a non-ML / off-topic
    hit for a query containing an ambiguous ML term. Used to filter obvious
    noise (Stephen Colbert talk-show papers when the user means ColBERT, etc.).
    """
    term = _detect_term(query)
    if not term:
        return False
    options = AMBIGUOUS_TERMS.get(term, [])
    if len(options) < 2:
        return False
    ml_sense = options[0]
    wrong_senses = options[1:]
    ml_keys = set(SENSE_KEYWORDS.get(ml_sense, ()))
    wrong_keys: set[str] = set()
    for s in wrong_senses:
        wrong_keys |= set(SENSE_KEYWORDS.get(s, ()))

    hay_text = f"{citation.get('title', '')} {citation.get('snippet', '')}".lower()
    hay_tokens = _tokens(hay_text)
    ml_hits = sum(1 for k in ml_keys if _word_hit(hay_tokens, hay_text, k))
    wrong_hits = sum(1 for k in wrong_keys if _word_hit(hay_tokens, hay_text, k))
    # Drop the result if it hits only wrong-sense keywords and zero ML ones.
    return wrong_hits > 0 and ml_hits == 0
