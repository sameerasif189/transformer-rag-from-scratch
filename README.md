# Assignment3-nlp
NLP Assignment 3 — Amazon Review Understanding & Explanation Generation
This project builds a three-part pipeline to classify Amazon product reviews and generate natural language explanations for those classifications.

What it does
Part A — Encoder (Multi-Task Classifier)
A Transformer encoder is trained from scratch to read a review and predict two things at once: the sentiment (negative / neutral / positive) and whether the review is short or detailed. It learns these simultaneously using a shared backbone with two separate output heads.
Part B — Retrieval Module
Once the encoder is trained, its internal representations (embeddings) are saved for every training review. At inference time, when given a new review, the system finds the 3 most similar reviews from the training set using cosine similarity. These are used as context for the next stage.
Part C — Decoder + RAG (Explanation Generator)
A Transformer decoder is trained from scratch to generate a short explanation for why a review has a given sentiment. During generation, the 3 retrieved reviews from Part B are prepended to the input — this is the Retrieval-Augmented Generation (RAG) step. The idea is that grounding the decoder in similar examples should produce more relevant explanations.

Results Summary
ComponentMetricResultEncoderTest Sentiment Accuracy82.04%EncoderTest Derived Feature Accuracy99.68%DecoderPerplexity (with retrieval)260.14DecoderPerplexity (without retrieval)74.95

Note: In this run, retrieval actually hurt perplexity rather than helping it. This is likely because the concatenated context made inputs too long for the small decoder to handle effectively. Further tuning is needed.


Dataset
Amazon product reviews split into 34,981 training, 7,495 validation, and 7,497 test samples, with a vocabulary of 30,000 tokens. All preprocessing (tokenization, vocabulary building, padding) is done on training data only to avoid leakage.
