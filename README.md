# Transformer + RAG from Scratch

A three-part NLP pipeline that classifies Amazon product reviews and generates
natural-language explanations for those classifications — with the Transformer encoder,
decoder, and retrieval module all implemented from scratch in PyTorch.

## What it does

**Part A — Encoder (multi-task classifier)**
A Transformer encoder trained from scratch reads a review and predicts two things at once:
sentiment (negative / neutral / positive) and whether the review is short or detailed,
using a shared backbone with two output heads.

**Part B — Retrieval module**
The encoder's internal embeddings are saved for every training review. At inference, a new
review retrieves its 3 most similar training reviews via cosine similarity to use as context.

**Part C — Decoder + RAG (explanation generator)**
A Transformer decoder trained from scratch generates a short explanation for a review's
sentiment. The 3 retrieved reviews are prepended to the input — the Retrieval-Augmented
Generation (RAG) step — to ground generation in similar examples.

## Results

| Component | Metric | Result |
|-----------|--------|--------|
| Encoder | Test sentiment accuracy | 82.04% |
| Encoder | Test derived-feature accuracy | 99.68% |
| Decoder | Perplexity (with retrieval) | 260.14 |
| Decoder | Perplexity (without retrieval) | 74.95 |

Note: in this run, retrieval increased perplexity rather than reducing it — likely because
the concatenated context made inputs too long for the small decoder. Further tuning needed.

## Dataset

Amazon product reviews split 70/15/15 into 34,981 train / 7,495 validation / 7,497 test
samples, vocabulary of 30,000 tokens. All preprocessing (tokenization, vocab building,
padding) is fit on training data only to avoid leakage.

## Run

See [RUN.md](RUN.md) for the full guide.

```bash
pip install torch
python transformer_rag_pipeline.py --data_dir "." --max_per_category 10000 \
    --epochs_encoder 2 --epochs_decoder 2 --k 3
```

## Tech

PyTorch, NumPy. Transformer encoder/decoder and retrieval implemented from scratch.
