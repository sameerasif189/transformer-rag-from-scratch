# NLP Assignment 3 Run Guide

This project now includes a full from-scratch pipeline in:

- `nlp_assignment3_pipeline.py`

It implements:
- Part A: encoder-only Transformer (multi-task: sentiment + derived feature)
- Part B: retrieval using encoder embeddings + cosine similarity top-k
- Part C: decoder-only Transformer for explanation generation
- Ablation: perplexity with retrieval vs without retrieval

## 1) Install Python and dependencies

Use Python 3.10+ and install:

```bash
pip install torch
```

## 2) Run the full pipeline

From this folder (`Dataset`):

```bash
python nlp_assignment3_pipeline.py --data_dir "." --max_per_category 10000 --epochs_encoder 2 --epochs_decoder 2 --k 3
```

## 3) Outputs

After running, you will get:

- `models/encoder.pt`
- `models/decoder.pt`
- `results/train_embeddings.pt`
- `results/metrics.json`

## 4) Notes for report

- **Derived feature used**: review length class (`short` vs `detailed`, threshold 20 tokens)
- **Sentiment mapping**:
  - 1-2: Negative
  - 3: Neutral
  - 4-5: Positive
- **Data split**: 70/15/15
- **Retrieval metric**: cosine similarity over normalized encoder embeddings
- **Ablation metric**: decoder perplexity with retrieval context vs no retrieval context

