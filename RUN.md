# Run Guide

A full from-scratch pipeline implemented in `transformer_rag_pipeline.py`:

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

From the data folder:

```bash
python transformer_rag_pipeline.py --data_dir "." --max_per_category 10000 --epochs_encoder 2 --epochs_decoder 2 --k 3
```

## 3) Outputs

- `models/encoder.pt`
- `models/decoder.pt`
- `results/train_embeddings.pt`
- `results/metrics.json`

## 4) Configuration notes

- **Derived feature**: review length class (`short` vs `detailed`, threshold 20 tokens)
- **Sentiment mapping**: 1-2 → Negative, 3 → Neutral, 4-5 → Positive
- **Data split**: 70/15/15
- **Retrieval metric**: cosine similarity over normalized encoder embeddings
- **Ablation metric**: decoder perplexity with vs without retrieval context
