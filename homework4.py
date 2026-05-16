# ============================================================
# Transformer from Scratch + Positional Encoding Ablation
# Reproduce core ideas of "Attention Is All You Need"
# Task: sequence reversal
#
# Experiment:
#   1. no positional encoding
#   2. sinusoidal positional encoding
#   3. learned absolute positional encoding
#
# Author: for PRML / Machine Learning homework
# ============================================================

import os
import math
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Global Configuration
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULT_DIR = "transformer_ablation_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# Special tokens
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

# Vocabulary:
# 0: PAD, 1: BOS, 2: EOS, 3...(VOCAB_SIZE-1): normal tokens
VOCAB_SIZE = 20

# Synthetic task settings
SEQ_LEN = 10
TRAIN_SAMPLES = 8000
VAL_SAMPLES = 1000
TEST_SAMPLES = 1000

# Model settings
D_MODEL = 64
N_HEADS = 4
D_FF = 256
N_LAYERS = 2
DROPOUT = 0.1
MAX_LEN = 64

# Training settings
BATCH_SIZE = 128
EPOCHS = 12
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# Positional encoding settings to compare
PE_TYPES = ["none", "sinusoidal", "learned"]


# ============================================================
# 2. Dataset
# ============================================================

class ReverseSequenceDataset(Dataset):
    """
    Synthetic sequence-to-sequence task.

    Given:
        src = [a1, a2, ..., aL]

    Target:
        output = [aL, ..., a2, a1, EOS]

    Decoder input:
        tgt_in = [BOS, aL, ..., a2, a1]

    Decoder output:
        tgt_out = [aL, ..., a2, a1, EOS]

    This task is suitable for positional encoding analysis because
    reversing a sequence requires knowing token order.
    """

    def __init__(self, num_samples, seq_len, vocab_size):
        super().__init__()

        self.src = []
        self.tgt_in = []
        self.tgt_out = []

        for _ in range(num_samples):
            seq = np.random.randint(3, vocab_size, size=(seq_len,), dtype=np.int64)
            rev = seq[::-1].copy()

            src = seq
            tgt_in = np.concatenate([[BOS_ID], rev])
            tgt_out = np.concatenate([rev, [EOS_ID]])

            self.src.append(src)
            self.tgt_in.append(tgt_in)
            self.tgt_out.append(tgt_out)

        self.src = torch.tensor(np.array(self.src), dtype=torch.long)
        self.tgt_in = torch.tensor(np.array(self.tgt_in), dtype=torch.long)
        self.tgt_out = torch.tensor(np.array(self.tgt_out), dtype=torch.long)

    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt_in[idx], self.tgt_out[idx]


def build_dataloaders():
    train_set = ReverseSequenceDataset(TRAIN_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    val_set = ReverseSequenceDataset(VAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    test_set = ReverseSequenceDataset(TEST_SAMPLES, SEQ_LEN, VOCAB_SIZE)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================
# 3. Positional Encoding
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Original sinusoidal positional encoding in Attention Is All You Need.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


class TokenPositionEmbedding(nn.Module):
    """
    Token embedding + positional encoding.

    pe_type:
        - "none": no position information
        - "sinusoidal": fixed sinusoidal positional encoding
        - "learned": learnable absolute positional encoding
    """

    def __init__(self, vocab_size, d_model, max_len, pe_type="sinusoidal", dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.pe_type = pe_type

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)

        if pe_type == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        elif pe_type == "learned":
            self.pos_emb = nn.Embedding(max_len, d_model)
        elif pe_type == "none":
            self.pos_emb = None
        else:
            raise ValueError(f"Unknown pe_type: {pe_type}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        """
        tokens shape: (batch_size, seq_len)
        """
        batch_size, seq_len = tokens.shape

        x = self.token_emb(tokens) * math.sqrt(self.d_model)

        if self.pe_type == "sinusoidal":
            x = x + self.pos_emb(x)

        elif self.pe_type == "learned":
            positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
            positions = positions.expand(batch_size, seq_len)
            x = x + self.pos_emb(positions)

        elif self.pe_type == "none":
            pass

        return self.dropout(x)


# ============================================================
# 4. Scaled Dot-Product Attention and Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        """
        query: (batch, len_q, d_model)
        key:   (batch, len_k, d_model)
        value: (batch, len_v, d_model)

        attn_mask:
            True means masked position.
            shape can be broadcast to (batch, n_heads, len_q, len_k)
        """

        batch_size = query.size(0)

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)

        output = self.w_o(context)

        return output, attn


# ============================================================
# 5. Feed Forward Network
# ============================================================

class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 6. Encoder and Decoder Layers
# ============================================================

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out, attn = self.self_attn(x, x, x, src_mask)

        # Residual connection + LayerNorm
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)

        # Residual connection + LayerNorm
        x = self.norm2(x + self.dropout2(ffn_out))

        return x, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        self_attn_out, self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        cross_attn_out, cross_attn = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x, self_attn, cross_attn


# ============================================================
# 7. Transformer Encoder-Decoder
# ============================================================

class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        dropout=0.1,
        max_len=64,
        pe_type="sinusoidal"
    ):
        super().__init__()

        self.pe_type = pe_type

        self.src_embedding = TokenPositionEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            pe_type=pe_type,
            dropout=dropout
        )

        self.tgt_embedding = TokenPositionEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            pe_type=pe_type,
            dropout=dropout
        )

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(d_model, vocab_size)

    def make_src_mask(self, src):
        """
        src shape: (batch, src_len)
        output mask shape: (batch, 1, 1, src_len)
        """
        return (src == PAD_ID).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        """
        tgt shape: (batch, tgt_len)
        output mask shape: (batch, 1, tgt_len, tgt_len)
        """
        batch_size, tgt_len = tgt.shape

        pad_mask = (tgt == PAD_ID).unsqueeze(1).unsqueeze(2)

        causal_mask = torch.triu(
            torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool),
            diagonal=1
        )

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        return pad_mask | causal_mask

    def encode(self, src):
        src_mask = self.make_src_mask(src)

        x = self.src_embedding(src)

        enc_attns = []

        for layer in self.encoder_layers:
            x, attn = layer(x, src_mask)
            enc_attns.append(attn)

        return x, src_mask, enc_attns

    def decode(self, tgt, memory, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)

        x = self.tgt_embedding(tgt)

        dec_self_attns = []
        dec_cross_attns = []

        for layer in self.decoder_layers:
            x, self_attn, cross_attn = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=src_mask
            )
            dec_self_attns.append(self_attn)
            dec_cross_attns.append(cross_attn)

        return x, dec_self_attns, dec_cross_attns

    def forward(self, src, tgt, return_attn=False):
        memory, src_mask, enc_attns = self.encode(src)
        dec_out, dec_self_attns, dec_cross_attns = self.decode(tgt, memory, src_mask)

        logits = self.output_proj(dec_out)

        if return_attn:
            return logits, {
                "encoder_self": enc_attns,
                "decoder_self": dec_self_attns,
                "decoder_cross": dec_cross_attns
            }

        return logits


# ============================================================
# 8. Training and Evaluation
# ============================================================

def token_accuracy(logits, target):
    """
    Token-level accuracy.
    """
    pred = logits.argmax(dim=-1)
    mask = target != PAD_ID

    correct = (pred == target) & mask
    total = mask.sum().item()

    if total == 0:
        return 0.0

    return correct.sum().item() / total


@torch.no_grad()
def greedy_decode(model, src, max_len):
    """
    Autoregressive decoding.

    src shape: (batch, src_len)
    output shape: (batch, max_len)
    """
    model.eval()

    batch_size = src.size(0)

    ys = torch.full(
        (batch_size, 1),
        BOS_ID,
        dtype=torch.long,
        device=src.device
    )

    for _ in range(max_len):
        logits = model(src, ys)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

    return ys[:, 1:]


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    total_seq = 0
    exact_match = 0

    for src, tgt_in, tgt_out in loader:
        src = src.to(DEVICE)
        tgt_in = tgt_in.to(DEVICE)
        tgt_out = tgt_out.to(DEVICE)

        logits = model(src, tgt_in)

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            tgt_out.reshape(-1)
        )

        batch_tokens = (tgt_out != PAD_ID).sum().item()

        pred = logits.argmax(dim=-1)
        correct = ((pred == tgt_out) & (tgt_out != PAD_ID)).sum().item()

        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        total_correct += correct

        generated = greedy_decode(model, src, max_len=tgt_out.size(1))
        exact_match += (generated == tgt_out).all(dim=1).sum().item()
        total_seq += src.size(0)

    avg_loss = total_loss / total_tokens
    tok_acc = total_correct / total_tokens
    seq_acc = exact_match / total_seq

    return avg_loss, tok_acc, seq_acc


def train_one_model(pe_type, train_loader, val_loader, test_loader):
    print("=" * 80)
    print(f"Start training Transformer with positional encoding: {pe_type}")
    print("=" * 80)

    model = TransformerSeq2Seq(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        max_len=MAX_LEN,
        pe_type=pe_type
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    history = []

    best_val_loss = float("inf")
    best_val_seq_acc = -1.0
    best_state = None

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_tokens = 0
        train_correct = 0

        for src, tgt_in, tgt_out in train_loader:
            src = src.to(DEVICE)
            tgt_in = tgt_in.to(DEVICE)
            tgt_out = tgt_out.to(DEVICE)

            optimizer.zero_grad()

            logits = model(src, tgt_in)

            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                tgt_out.reshape(-1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            with torch.no_grad():
                batch_tokens = (tgt_out != PAD_ID).sum().item()
                pred = logits.argmax(dim=-1)
                correct = ((pred == tgt_out) & (tgt_out != PAD_ID)).sum().item()

            train_loss_sum += loss.item() * batch_tokens
            train_tokens += batch_tokens
            train_correct += correct

        train_loss = train_loss_sum / train_tokens
        train_tok_acc = train_correct / train_tokens

        val_loss, val_tok_acc, val_seq_acc = evaluate(model, val_loader, criterion)

        history.append({
            "epoch": epoch,
            "pe_type": pe_type,
            "train_loss": train_loss,
            "train_token_acc": train_tok_acc,
            "val_loss": val_loss,
            "val_token_acc": val_tok_acc,
            "val_seq_acc": val_seq_acc
        })

        print(
            f"[{pe_type}] Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f}, train_tok_acc={train_tok_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_tok_acc={val_tok_acc:.4f}, "
            f"val_seq_acc={val_seq_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_seq_acc = val_seq_acc
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_tok_acc, test_seq_acc = evaluate(model, test_loader, criterion)

    elapsed = time.time() - start_time

    result = {
        "pe_type": pe_type,
        "test_loss": test_loss,
        "test_token_acc": test_tok_acc,
        "test_seq_acc": test_seq_acc,
        "best_val_loss": best_val_loss,
        "best_val_seq_acc": best_val_seq_acc,
        "time_sec": elapsed,
        "params": sum(p.numel() for p in model.parameters())
    }

    print("-" * 80)
    print(f"Final test result for {pe_type}:")
    print(result)
    print("-" * 80)

    return model, pd.DataFrame(history), result


# ============================================================
# 9. Visualization
# ============================================================

def plot_training_curves(all_histories):
    plt.figure(figsize=(10, 6))

    for pe_type, hist in all_histories.items():
        plt.plot(hist["epoch"], hist["val_seq_acc"], marker="o", label=f"{pe_type}")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Exact Sequence Accuracy")
    plt.title("Validation Sequence Accuracy under Different Positional Encodings")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "fig1_validation_sequence_accuracy.png")
    plt.savefig(path, dpi=300)
    plt.show()


def plot_test_bar(results_df):
    plt.figure(figsize=(9, 5))

    plt.bar(results_df["pe_type"], results_df["test_seq_acc"])

    plt.xlabel("Positional Encoding Type")
    plt.ylabel("Test Exact Sequence Accuracy")
    plt.title("Test Accuracy Comparison of Positional Encoding Methods")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "fig2_test_sequence_accuracy.png")
    plt.savefig(path, dpi=300)
    plt.show()


@torch.no_grad()
def plot_cross_attention_heatmap(model, loader, pe_type):
    """
    Plot cross-attention heatmap of the last decoder layer.
    The heatmap is averaged over heads for one sample.
    """
    model.eval()

    src, tgt_in, tgt_out = next(iter(loader))

    src = src[:1].to(DEVICE)
    tgt_in = tgt_in[:1].to(DEVICE)
    tgt_out = tgt_out[:1].to(DEVICE)

    logits, attns = model(src, tgt_in, return_attn=True)

    cross_attn = attns["decoder_cross"][-1]  # last decoder layer
    cross_attn = cross_attn[0].mean(dim=0).detach().cpu().numpy()
    # shape: (target_len, source_len)

    src_tokens = src[0].detach().cpu().numpy().tolist()
    tgt_tokens = tgt_out[0].detach().cpu().numpy().tolist()

    plt.figure(figsize=(8, 6))
    plt.imshow(cross_attn, aspect="auto", cmap="viridis")
    plt.colorbar(label="Attention Weight")

    plt.xticks(range(len(src_tokens)), src_tokens)
    plt.yticks(range(len(tgt_tokens)), tgt_tokens)

    plt.xlabel("Source Tokens")
    plt.ylabel("Target Tokens")
    plt.title(f"Decoder Cross-Attention Heatmap ({pe_type})")

    plt.tight_layout()

    path = os.path.join(RESULT_DIR, f"fig3_cross_attention_{pe_type}.png")
    plt.savefig(path, dpi=300)
    plt.show()


def save_experiment_summary(results_df, all_histories):
    result_path = os.path.join(RESULT_DIR, "positional_encoding_ablation_results.csv")
    results_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    hist_df = pd.concat(all_histories.values(), axis=0)
    hist_path = os.path.join(RESULT_DIR, "training_history.csv")
    hist_df.to_csv(hist_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Saved results:")
    print(result_path)
    print(hist_path)
    print("=" * 80)


# ============================================================
# 10. Main Function
# ============================================================

def main():
    print("=" * 80)
    print("Transformer Positional Encoding Ablation Experiment")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    train_loader, val_loader, test_loader = build_dataloaders()

    all_histories = {}
    all_results = []
    trained_models = {}

    for pe_type in PE_TYPES:
        model, history, result = train_one_model(
            pe_type=pe_type,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )

        all_histories[pe_type] = history
        all_results.append(result)
        trained_models[pe_type] = model

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="test_seq_acc", ascending=False)

    print("=" * 80)
    print("Ablation Result Summary:")
    print(results_df)
    print("=" * 80)

    save_experiment_summary(results_df, all_histories)

    plot_training_curves(all_histories)
    plot_test_bar(results_df)

    # Plot attention heatmap for the best model
    best_pe = results_df.iloc[0]["pe_type"]
    plot_cross_attention_heatmap(trained_models[best_pe], test_loader, best_pe)

    print("=" * 80)
    print("Experiment finished.")
    print(f"All outputs are saved in: {RESULT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()