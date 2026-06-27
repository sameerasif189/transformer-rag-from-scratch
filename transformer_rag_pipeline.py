import argparse
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s\.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return text.split()


def sentiment_from_rating(rating: float) -> int:
    if rating <= 2:
        return 0  # negative
    if rating == 3:
        return 1  # neutral
    return 2  # positive


def derived_feature_from_text(text: str, threshold: int = 20) -> int:
    return 1 if len(tokenize(text)) >= threshold else 0


def explanation_target(sentiment_id: int, length_flag: int) -> str:
    s_map = {0: "negative", 1: "neutral", 2: "positive"}
    l_map = {0: "short", 1: "detailed"}
    return (
        f"the review appears {s_map[sentiment_id]} because the wording and tone suggest "
        f"a {l_map[length_flag]} opinion about product quality."
    )


def load_category_file(path: str, max_samples: int) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines; Amazon dumps can contain occasional bad records.
                continue
            text = obj.get("reviewText", "").strip()
            rating = obj.get("overall", None)
            if text and rating is not None:
                rows.append({"review_text": text, "rating": float(rating)})
    return rows


def gather_dataset(
    root_dir: str,
    max_per_category: int = 12000,
    min_categories: int = 3,
    category_dirs: List[str] = None,
) -> List[dict]:
    category_files = []
    if category_dirs:
        for d in category_dirs:
            if os.path.isfile(d) and d.endswith(".json"):
                category_files.append(d)
                continue
            if not os.path.isdir(d):
                continue
            try:
                for name in os.listdir(d):
                    if name.endswith(".json"):
                        category_files.append(os.path.join(d, name))
            except (PermissionError, OSError):
                continue
    else:
        for child in os.listdir(root_dir):
            full = os.path.join(root_dir, child)
            if not os.path.isdir(full):
                continue
            try:
                for name in os.listdir(full):
                    if name.endswith(".json"):
                        category_files.append(os.path.join(full, name))
            except (PermissionError, OSError):
                # Skip protected/system directories when running from broad notebook paths.
                continue

    if len(category_files) < min_categories:
        raise RuntimeError(
            f"Need at least {min_categories} category JSON files, found {len(category_files)}. "
            f"Check data_dir='{root_dir}' and pass category_dirs explicitly in notebook."
        )

    data = []
    for path in sorted(category_files)[:max(min_categories, len(category_files))]:
        category_name = os.path.basename(path).replace(".json", "")
        rows = load_category_file(path, max_per_category)
        for r in rows:
            r["category"] = category_name
        data.extend(rows)
    random.shuffle(data)
    return data


def split_data(data: List[dict], train_ratio=0.7, val_ratio=0.15):
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    return train, val, test


SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]


def build_vocab(samples: List[dict], min_freq: int = 2, max_vocab: int = 30000) -> Dict[str, int]:
    counter = Counter()
    for s in samples:
        counter.update(tokenize(clean_text(s["review_text"])))
        counter.update(tokenize(clean_text(s["explanation"])))
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if tok in vocab:
            continue
        if len(vocab) >= max_vocab:
            break
        vocab[tok] = len(vocab)
    return vocab


def encode_tokens(tokens: List[str], vocab: Dict[str, int], max_len: int, add_bos_eos: bool = False):
    ids = []
    if add_bos_eos:
        ids.append(vocab["<bos>"])
    for t in tokens:
        ids.append(vocab.get(t, vocab["<unk>"]))
    if add_bos_eos:
        ids.append(vocab["<eos>"])
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids.extend([vocab["<pad>"]] * (max_len - len(ids)))
    return ids


class ReviewsDataset(Dataset):
    def __init__(self, rows: List[dict], vocab: Dict[str, int], max_review_len: int, max_prompt_len: int, max_out_len: int):
        self.rows = rows
        self.vocab = vocab
        self.max_review_len = max_review_len
        self.max_prompt_len = max_prompt_len
        self.max_out_len = max_out_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        review_tokens = tokenize(clean_text(r["review_text"]))
        review_ids = encode_tokens(review_tokens, self.vocab, self.max_review_len, add_bos_eos=False)

        sent = r["sentiment"]
        feat = r["derived"]
        retrieved = r.get("retrieved_text", "")
        prompt = (
            f"review: {clean_text(r['review_text'])} <sep> sentiment: {sent} "
            f"<sep> feature: {feat} <sep> retrieved: {clean_text(retrieved)}"
        )
        prompt_ids = encode_tokens(tokenize(prompt), self.vocab, self.max_prompt_len, add_bos_eos=True)
        expl_ids = encode_tokens(tokenize(clean_text(r["explanation"])), self.vocab, self.max_out_len, add_bos_eos=True)

        return {
            "review_ids": torch.tensor(review_ids, dtype=torch.long),
            "sentiment": torch.tensor(sent, dtype=torch.long),
            "derived": torch.tensor(feat, dtype=torch.long),
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "target_ids": torch.tensor(expl_ids, dtype=torch.long),
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ff(x))
        return x


class EncoderOnlyModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.sent_head = nn.Linear(d_model, 3)
        self.derived_head = nn.Linear(d_model, 2)

    def forward(self, ids, pad_id: int):
        mask = (ids != pad_id).unsqueeze(1).unsqueeze(2)
        x = self.embed(ids)
        x = self.pos(x)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        pooled = x[:, 0, :]
        sent_logits = self.sent_head(pooled)
        derived_logits = self.derived_head(pooled)
        return sent_logits, derived_logits, pooled


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, causal_mask):
        x = self.norm1(x + self.attn(x, mask=causal_mask))
        x = self.norm2(x + self.ff(x))
        return x


class DecoderOnlyModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        b, t = ids.shape
        x = self.embed(ids)
        x = self.pos(x)
        x = self.drop(x)
        causal = torch.tril(torch.ones(t, t, device=ids.device)).unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, causal_mask=causal)
        return self.out(x)


def accuracy(pred, y):
    return (pred.argmax(dim=-1) == y).float().mean().item()


@dataclass
class TrainStats:
    losses: List[float]
    sent_acc: List[float]
    derived_acc: List[float]


def train_encoder(model, train_loader, val_loader, device, pad_id, epochs=3, lr=3e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    stats = TrainStats([], [], [])
    best_val = 0.0
    os.makedirs("models", exist_ok=True)
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        total_sa = 0.0
        total_da = 0.0
        n = 0
        for batch in train_loader:
            ids = batch["review_ids"].to(device)
            s = batch["sentiment"].to(device)
            d = batch["derived"].to(device)
            opt.zero_grad()
            s_logit, d_logit, _ = model(ids, pad_id)
            loss = ce(s_logit, s) + ce(d_logit, d)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_sa += accuracy(s_logit, s)
            total_da += accuracy(d_logit, d)
            n += 1
        stats.losses.append(total_loss / max(n, 1))
        stats.sent_acc.append(total_sa / max(n, 1))
        stats.derived_acc.append(total_da / max(n, 1))

        val_sa, val_da = evaluate_encoder(model, val_loader, device, pad_id)
        val_score = (val_sa + val_da) / 2
        if val_score > best_val:
            best_val = val_score
            torch.save(model.state_dict(), os.path.join("models", "encoder.pt"))
        print(f"[Encoder] Epoch {ep+1}/{epochs} loss={stats.losses[-1]:.4f} train_sa={stats.sent_acc[-1]:.4f} train_da={stats.derived_acc[-1]:.4f} val_sa={val_sa:.4f} val_da={val_da:.4f}")
    return stats


def evaluate_encoder(model, loader, device, pad_id):
    model.eval()
    sa = 0.0
    da = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["review_ids"].to(device)
            s = batch["sentiment"].to(device)
            d = batch["derived"].to(device)
            s_logit, d_logit, _ = model(ids, pad_id)
            sa += accuracy(s_logit, s)
            da += accuracy(d_logit, d)
            n += 1
    return sa / max(n, 1), da / max(n, 1)


def encode_all_embeddings(model, loader, device, pad_id):
    model.eval()
    embs = []
    rows = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["review_ids"].to(device)
            _, _, pooled = model(ids, pad_id)
            embs.append(pooled.cpu())
            for i in range(ids.size(0)):
                rows.append(ids[i].cpu())
    return torch.cat(embs, dim=0), rows


def retrieve_top_k(query_emb: torch.Tensor, corpus_emb: torch.Tensor, k=3):
    q = F.normalize(query_emb, dim=-1)
    c = F.normalize(corpus_emb, dim=-1)
    sims = q @ c.T
    vals, idx = torch.topk(sims, k=min(k, c.size(0)), dim=-1)
    return vals, idx


def ids_to_text(ids: List[int], inv_vocab: Dict[int, str]):
    toks = []
    for i in ids:
        tok = inv_vocab.get(int(i), "<unk>")
        if tok in ("<pad>", "<bos>", "<eos>"):
            continue
        toks.append(tok)
    return " ".join(toks)


def attach_retrieval_text(train_rows: List[dict], test_rows: List[dict], train_emb: torch.Tensor, test_emb: torch.Tensor, k: int):
    for r in train_rows:
        r["retrieved_text"] = ""
    vals, idx = retrieve_top_k(test_emb, train_emb, k=k)
    for i, r in enumerate(test_rows):
        snippets = []
        for j in idx[i].tolist():
            snippets.append(train_rows[j]["review_text"])
        r["retrieved_text"] = " ".join(snippets)
    return vals, idx


def train_decoder(model, loader, device, pad_id, epochs=3, lr=3e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(ignore_index=pad_id)
    os.makedirs("models", exist_ok=True)
    best = 1e9
    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            prompt = batch["prompt_ids"].to(device)
            target = batch["target_ids"].to(device)
            inp = torch.cat([prompt, target[:, :-1]], dim=1)
            gold = torch.cat([prompt[:, 1:], target], dim=1)
            logits = model(inp)
            loss = ce(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        avg = total / max(n, 1)
        print(f"[Decoder] Epoch {ep+1}/{epochs} loss={avg:.4f}")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join("models", "decoder.pt"))


def perplexity(model, loader, device, pad_id):
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            prompt = batch["prompt_ids"].to(device)
            target = batch["target_ids"].to(device)
            inp = torch.cat([prompt, target[:, :-1]], dim=1)
            gold = torch.cat([prompt[:, 1:], target], dim=1)
            logits = model(inp)
            loss = ce(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            total_loss += loss.item()
            total_tokens += (gold != pad_id).sum().item()
    return math.exp(total_loss / max(total_tokens, 1))


def generate_text(model, prompt_ids, max_new_tokens, bos_id, eos_id, device):
    model.eval()
    seq = prompt_ids.clone().to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(seq)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_id], dim=1)
            if next_id.item() == eos_id:
                break
    return seq[0].cpu().tolist()


def prepare_rows(data: List[dict]) -> List[dict]:
    rows = []
    for d in data:
        txt = clean_text(d["review_text"])
        sent = sentiment_from_rating(d["rating"])
        feat = derived_feature_from_text(txt)
        rows.append(
            {
                "review_text": txt,
                "rating": d["rating"],
                "category": d["category"],
                "sentiment": sent,
                "derived": feat,
                "explanation": explanation_target(sent, feat),
            }
        )
    return rows


def save_metrics(path: str, metrics: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def run_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    raw = gather_dataset(
        args.data_dir,
        max_per_category=args.max_per_category,
        category_dirs=getattr(args, "category_dirs", None),
    )
    rows = prepare_rows(raw)
    train_rows, val_rows, test_rows = split_data(rows)
    vocab = build_vocab(train_rows, min_freq=2, max_vocab=30000)
    inv_vocab = {v: k for k, v in vocab.items()}
    pad_id = vocab["<pad>"]

    train_ds = ReviewsDataset(train_rows, vocab, args.max_review_len, args.max_prompt_len, args.max_out_len)
    val_ds = ReviewsDataset(val_rows, vocab, args.max_review_len, args.max_prompt_len, args.max_out_len)
    test_ds = ReviewsDataset(test_rows, vocab, args.max_review_len, args.max_prompt_len, args.max_out_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    encoder = EncoderOnlyModel(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_len=args.max_review_len + 5,
    ).to(device)
    enc_stats = train_encoder(encoder, train_loader, val_loader, device, pad_id, epochs=args.epochs_encoder)
    encoder.load_state_dict(torch.load(os.path.join("models", "encoder.pt"), map_location=device))
    test_sa, test_da = evaluate_encoder(encoder, test_loader, device, pad_id)

    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    train_emb, _ = encode_all_embeddings(encoder, train_eval_loader, device, pad_id)
    test_emb, _ = encode_all_embeddings(encoder, test_loader, device, pad_id)
    os.makedirs("results", exist_ok=True)
    torch.save(train_emb, os.path.join("results", "train_embeddings.pt"))

    _, _ = attach_retrieval_text(train_rows, test_rows, train_emb, test_emb, k=args.k)
    for r in train_rows:
        r["retrieved_text"] = ""

    train_ds_dec = ReviewsDataset(train_rows, vocab, args.max_review_len, args.max_prompt_len, args.max_out_len)
    test_ds_dec = ReviewsDataset(test_rows, vocab, args.max_review_len, args.max_prompt_len, args.max_out_len)
    train_loader_dec = DataLoader(train_ds_dec, batch_size=args.batch_size, shuffle=True)
    test_loader_dec = DataLoader(test_ds_dec, batch_size=args.batch_size, shuffle=False)

    decoder = DecoderOnlyModel(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_len=args.max_prompt_len + args.max_out_len + 10,
    ).to(device)
    train_decoder(decoder, train_loader_dec, device, pad_id, epochs=args.epochs_decoder)
    decoder.load_state_dict(torch.load(os.path.join("models", "decoder.pt"), map_location=device))
    ppl_with_retrieval = perplexity(decoder, test_loader_dec, device, pad_id)

    # Ablation: no retrieval context
    test_no_ret = [dict(r, retrieved_text="") for r in test_rows]
    test_no_ret_ds = ReviewsDataset(test_no_ret, vocab, args.max_review_len, args.max_prompt_len, args.max_out_len)
    test_no_ret_loader = DataLoader(test_no_ret_ds, batch_size=args.batch_size, shuffle=False)
    ppl_without_retrieval = perplexity(decoder, test_no_ret_loader, device, pad_id)

    samples = []
    bos = vocab["<bos>"]
    eos = vocab["<eos>"]
    for i in range(min(5, len(test_rows))):
        prompt = (
            f"review: {test_rows[i]['review_text']} <sep> sentiment: {test_rows[i]['sentiment']} "
            f"<sep> feature: {test_rows[i]['derived']} <sep> retrieved: {test_rows[i]['retrieved_text']}"
        )
        prompt_ids = encode_tokens(tokenize(prompt), vocab, args.max_prompt_len, add_bos_eos=True)
        seq = torch.tensor([prompt_ids], dtype=torch.long)
        out_ids = generate_text(decoder, seq, max_new_tokens=30, bos_id=bos, eos_id=eos, device=device)
        samples.append(
            {
                "review": test_rows[i]["review_text"][:240],
                "sentiment": test_rows[i]["sentiment"],
                "derived": test_rows[i]["derived"],
                "generated": ids_to_text(out_ids, inv_vocab),
                "reference": test_rows[i]["explanation"],
            }
        )

    metrics = {
        "data_size": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "vocab_size": len(vocab),
        },
        "encoder": {
            "train_losses": enc_stats.losses,
            "train_sent_acc": enc_stats.sent_acc,
            "train_derived_acc": enc_stats.derived_acc,
            "test_sent_acc": test_sa,
            "test_derived_acc": test_da,
        },
        "retrieval": {"k": args.k, "embedding_file": "results/train_embeddings.pt"},
        "decoder": {
            "perplexity_with_retrieval": ppl_with_retrieval,
            "perplexity_without_retrieval": ppl_without_retrieval,
        },
        "generated_samples": samples,
    }
    save_metrics(os.path.join("results", "metrics.json"), metrics)
    print("Done. Saved models to models/ and outputs to results/metrics.json")
    return metrics


def run_in_notebook(
    data_dir=".",
    category_dirs=None,
    max_per_category=10000,
    batch_size=32,
    epochs_encoder=10,
    epochs_decoder=10,
    max_review_len=128,
    max_prompt_len=180,
    max_out_len=40,
    d_model=128,
    n_heads=4,
    d_ff=256,
    n_layers=2,
    k=3,
):
    """
    Notebook helper:
    Example:
        from transformer_rag_pipeline import run_in_notebook
        metrics = run_in_notebook(data_dir='.', epochs_encoder=10, epochs_decoder=10)
    """
    args = argparse.Namespace(
        data_dir=data_dir,
        category_dirs=category_dirs,
        max_per_category=max_per_category,
        batch_size=batch_size,
        epochs_encoder=epochs_encoder,
        epochs_decoder=epochs_decoder,
        max_review_len=max_review_len,
        max_prompt_len=max_prompt_len,
        max_out_len=max_out_len,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        k=k,
    )
    return run_pipeline(args)


def main():
    parser = argparse.ArgumentParser(description="Transformer + RAG from scratch")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--category_dirs", type=str, nargs="*", default=None)
    parser.add_argument("--max_per_category", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_encoder", type=int, default=10)
    parser.add_argument("--epochs_decoder", type=int, default=10)
    parser.add_argument("--max_review_len", type=int, default=128)
    parser.add_argument("--max_prompt_len", type=int, default=180)
    parser.add_argument("--max_out_len", type=int, default=40)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--k", type=int, default=3)
    # In notebooks, IPython injects extra args like "-f <kernel.json>".
    # parse_known_args keeps CLI behavior while ignoring unknown kernel args.
    args, _ = parser.parse_known_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
