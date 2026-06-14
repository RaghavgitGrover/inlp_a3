"""
INLP Assignment 3 — Task 2 (v11)
Next-Word Prediction (SSM) + Masked Language Modeling (Bi-LSTM)

CHANGELOG (v11):
1. FROZEN S4D MEMORY: Explicitly froze the `A` matrix (`log_A_real`, `A_imag`). 
   If `A` is updated heavily on small datasets, the model destroys its continuous-time 
   memory prior and catastrophically overfits (as seen in previous runs).
2. AGGRESSIVE REGULARIZATION: Increased dropout to 0.4 and pruned rare vocab words 
   (min_freq=3) to prevent the model from memorizing the training set.
3. OPTIMIZED LR: Scaled learning rates down to 5e-4 to find broader, more generalizable 
   minima rather than sharp overfit minima.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "/kaggle/input/datasets/raghav1grover1/inlpa3/plain.txt"
OUTPUT_DIR = "outputs"
SEED = 42

WANDB_API_KEY = "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"
WANDB_PROJECT = "INLP-A3-Task2"
WANDB_ENTITY = None

# ── SSM (S4D) ──
SSM_D_MODEL = 128
SSM_D_STATE = 64
SSM_N_LAYERS = 3
SSM_SEQ_LEN = 128
SSM_EPOCHS = 80
SSM_LR = 5e-4           # Lowered to prevent jumping into sharp overfit minima
SSM_WARMUP = 8
SSM_BATCH = 64
SSM_DROPOUT = 0.4       # Aggressive dropout to combat overfitting
SSM_WEIGHT_DECAY = 0.05
SSM_LABEL_SMOOTH = 0.1
SSM_PATIENCE = 15

# ── Custom Bi-LSTM ──
BILSTM_EMBED = 128
BILSTM_HIDDEN = 256
BILSTM_LAYERS = 2
BILSTM_SEQ_LEN = 128
BILSTM_STRIDE = 64 
BILSTM_EPOCHS = 80
BILSTM_LR = 5e-4
BILSTM_WARMUP = 8
BILSTM_BATCH = 64
BILSTM_DROPOUT = 0.4
BILSTM_WEIGHT_DECAY = 0.05
BILSTM_LABEL_SMOOTH = 0.1
BILSTM_PATIENCE = 15

MASK_PROB = 0.15
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, math, json, time, random, logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import wandb
    wandb.login(key=WANDB_API_KEY, relogin=True)
    WANDB_AVAILABLE = True
    print("[INFO] wandb login successful.")
except Exception as e:
    WANDB_AVAILABLE = False
    print(f"[WARN] wandb unavailable: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def setup_dirs():
    dirs = {"base": Path(OUTPUT_DIR)}
    for name in ("logs", "plots", "results", "models"):
        dirs[name] = Path(OUTPUT_DIR) / name
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)
    return dirs

def setup_logging(log_path):
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"),
                  logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)

def plot_curves(train_vals, val_vals, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(train_vals) + 1)
    ax.plot(epochs, train_vals, label=f"Train {ylabel}", linewidth=2)
    ax.plot(epochs, val_vals, label=f"Val {ylabel}", linewidth=2)
    ax.set_title(title, fontsize=13); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close(fig)

def perplexity(loss): return math.exp(min(loss, 30))

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    base_lr = optimizer.param_groups[0]["lr"]
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(min_lr / base_lr, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def create_optimizer(model, lr, weight_decay):
    """
    Safely separates parameters to ensure SSM state parameters, biases, 
    and LayerNorms DO NOT receive weight decay, preventing catastrophic overfitting.
    """
    decay = []
    no_decay =[]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude memory states and norm/bias from weight decay
        if any(k in name for k in['log_dt', 'B', 'C', 'D', 'bias', 'norm']):
            no_decay.append(param)
        else:
            decay.append(param)
            
    optim_groups =[
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=lr)

# ─────────────────────────────────────────────────────────────────────────────
# VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────
class Vocabulary:
    PAD, UNK, MASK, EOS = "<PAD>", "<UNK>", "<MASK>", "<EOS>"
    def __init__(self, min_freq=3):  # Increased min_freq to prune rare words & prevent memorization
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        for i, tok in enumerate([self.PAD, self.UNK, self.MASK, self.EOS]):
            self.word2idx[tok] = i; self.idx2word[i] = tok

    def build(self, tokens: List[str]):
        freq: Dict[str, int] = {}
        for w in tokens: freq[w] = freq.get(w, 0) + 1
        idx = len(self.word2idx)
        for w in sorted(freq):
            if freq[w] >= self.min_freq and w not in self.word2idx:
                self.word2idx[w] = idx; self.idx2word[idx] = w; idx += 1
        print(f"[Vocab] size={len(self.word2idx)}")

    def encode(self, tokens):
        unk = self.word2idx[self.UNK]
        return[self.word2idx.get(t, unk) for t in tokens]

    def decode(self, ids):
        return [self.idx2word.get(i, self.UNK) for i in ids]

    def __len__(self): return len(self.word2idx)
    @property
    def pad_idx(self): return self.word2idx[self.PAD]
    @property
    def mask_idx(self): return self.word2idx[self.MASK]
    @property
    def eos_idx(self): return self.word2idx[self.EOS]

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_and_tokenize(path):
    sentences =[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().lower().split()
            if tokens: sentences.append(tokens)
    print(f"[Data] {len(sentences)} sentences from {path}")
    return sentences

def sentences_to_stream(sentences: List[List[str]], eos_token: str) -> List[str]:
    stream =[]
    for sent in sentences:
        stream.extend(sent)
        stream.append(eos_token)
    return stream

def split_sentences(sentences):
    random.shuffle(sentences)
    n = len(sentences)
    t = int(n * TRAIN_SPLIT); v = int(n * (TRAIN_SPLIT + VAL_SPLIT))
    return sentences[:t], sentences[t:v], sentences[v:]

class NWPStreamDataset(Dataset):
    def __init__(self, sentences, vocab, seq_len):
        stream = sentences_to_stream(sentences, vocab.EOS)
        ids = vocab.encode(stream)
        self.samples =[]
        for i in range(0, len(ids) - seq_len, seq_len):
            x = ids[i : i + seq_len]
            y = ids[i + 1 : i + seq_len + 1]
            if len(x) == seq_len and len(y) == seq_len:
                self.samples.append((x, y))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class MLMStreamDataset(Dataset):
    def __init__(self, sentences, vocab, seq_len, stride, mask_prob):
        stream = sentences_to_stream(sentences, vocab.EOS)
        ids = vocab.encode(stream)
        self.vocab = vocab; self.mask_prob = mask_prob
        self.samples =[]
        for i in range(0, len(ids) - seq_len + 1, stride):
            chunk = ids[i:i+seq_len]
            if len(chunk) == seq_len:
                self.samples.append(chunk)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        ids = self.samples[idx][:]
        inp, lbl = [],[]
        for tok in ids:
            if random.random() < self.mask_prob:
                inp.append(self.vocab.mask_idx); lbl.append(tok)
            else:
                inp.append(tok); lbl.append(-100)
        return (torch.tensor(inp, dtype=torch.long),
                torch.tensor(lbl, dtype=torch.long))

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing; self.vocab_size = vocab_size
        self.ignore_index = ignore_index
    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
            mask = (targets == self.ignore_index)
            valid = targets.clone(); valid[mask] = 0
            smooth.scatter_(1, valid.unsqueeze(1), 1.0 - self.smoothing)
            smooth[mask] = 0.0
        loss = -(smooth * log_probs).sum(dim=-1)
        loss[mask] = 0.0
        return loss.sum() / (~mask).sum().clamp(min=1)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — S4D (Diagonal State Space for NWP)
# ─────────────────────────────────────────────────────────────────────────────
class S4D(nn.Module):
    """
    S4D: Diagonal State Space Model.
    Mathematically equivalent to the diagonalized HiPPO formulation.
    """
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.h = d_model
        self.n = d_state

        # S4D-Lin Initialization
        log_dt = torch.rand(self.h) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        self.log_dt = nn.Parameter(log_dt)

        A_real = torch.full((self.h, self.n), -0.5)
        A_imag = math.pi * (torch.arange(self.n) + 0.5).unsqueeze(0).repeat(self.h, 1)
        
        # FREEZE THE MEMORY MATRIX 'A'
        # Crucial to prevent catastrophic overfitting on small/medium datasets
        self.log_A_real = nn.Parameter(torch.log(torch.abs(A_real)), requires_grad=False)
        self.A_imag = nn.Parameter(A_imag, requires_grad=False)

        self.B = nn.Parameter(torch.randn(self.h, self.n, dtype=torch.cfloat) * 0.1)
        self.C = nn.Parameter(torch.randn(self.h, self.n, dtype=torch.cfloat) * 0.1)
        self.D = nn.Parameter(torch.randn(self.h) * 0.1)

    def forward(self, x):
        b, l, h = x.shape

        dt = torch.exp(self.log_dt) 
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag 

        dtA = A * dt.unsqueeze(-1) 
        B_discrete = (torch.exp(dtA) - 1.0) / A * self.B 

        t = torch.arange(l, device=x.device) 
        dtA_t = dtA.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0) 
        powers = torch.exp(dtA_t) 

        K = torch.einsum('hn, hnl -> hl', self.C * B_discrete, powers).real 

        x_transpose = x.transpose(1, 2) 
        
        K_f = torch.fft.rfft(K, n=2*l) 
        x_f = torch.fft.rfft(x_transpose, n=2*l) 
        
        y = torch.fft.irfft(x_f * K_f, n=2*l)[..., :l] 
        y = y.transpose(1, 2) 
        return y + x * self.D

class GatedSSMBlock(nn.Module):
    def __init__(self, d_model, d_state, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.expand = nn.Linear(d_model, 2 * d_model, bias=False)
        self.ssm = S4D(d_model, d_state)
        self.contract = nn.Linear(d_model, d_model, bias=False)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model), nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        v, gate = self.expand(x_norm).chunk(2, dim=-1)
        v_ssm = self.ssm(v)
        out = self.contract(v_ssm * F.silu(gate))
        x = x + self.dropout(out)
        
        x = x + self.ffn(self.norm2(x))
        return x

class SSMLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, n_layers, dropout, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.edrop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GatedSSMBlock(d_model, d_state, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=True)
        self.head.weight = self.embed.weight
        
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        e = self.edrop(self.embed(x))
        for blk in self.blocks: 
            e = blk(e)
        return self.head(self.norm(e)) 

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — Custom Bi-LSTM (No nn.LSTM to strictly adhere to guidelines)
# ─────────────────────────────────────────────────────────────────────────────
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        self.W_ih_f = nn.Linear(input_size, 4 * hidden_size)
        self.W_hh_f = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        
        if bidirectional:
            self.W_ih_b = nn.Linear(input_size, 4 * hidden_size)
            self.W_hh_b = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
            
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
                if 'W_ih' in name:
                    with torch.no_grad():
                        p[self.hidden_size:2*self.hidden_size].fill_(1.0)

    def forward(self, x):
        B, T, _ = x.shape
        device = x.device
        
        gates_x_f = self.W_ih_f(x)
        h_f = torch.zeros(B, self.hidden_size, device=device)
        c_f = torch.zeros(B, self.hidden_size, device=device)
        out_f =[]
        
        for t in range(T):
            gates = gates_x_f[:, t, :] + self.W_hh_f(h_f)
            i, f, g, o = gates.chunk(4, dim=-1)
            c_f = torch.sigmoid(f) * c_f + torch.sigmoid(i) * torch.tanh(g)
            h_f = torch.sigmoid(o) * torch.tanh(c_f)
            out_f.append(h_f)
        out_f = torch.stack(out_f, dim=1)
        
        if self.bidirectional:
            gates_x_b = self.W_ih_b(x)
            h_b = torch.zeros(B, self.hidden_size, device=device)
            c_b = torch.zeros(B, self.hidden_size, device=device)
            out_b =[]
            
            for t in range(T - 1, -1, -1):
                gates = gates_x_b[:, t, :] + self.W_hh_b(h_b)
                i, f, g, o = gates.chunk(4, dim=-1)
                c_b = torch.sigmoid(f) * c_b + torch.sigmoid(i) * torch.tanh(g)
                h_b = torch.sigmoid(o) * torch.tanh(c_b)
                out_b.append(h_b)
            out_b = torch.stack(out_b[::-1], dim=1)
            return torch.cat([out_f, out_b], dim=-1)
            
        return out_f

class CustomMultiLayerBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.num_layers = num_layers
        
        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size * 2
            self.layers.append(CustomLSTM(in_sz, hidden_size, bidirectional=True))
            if i < num_layers - 1:
                self.dropouts.append(nn.Dropout(dropout))
                
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = self.dropouts[i](x)
        return x

class BiLSTMMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = CustomMultiLayerBiLSTM(embed_dim, hidden_dim, num_layers=n_layers, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(2 * hidden_dim)
        self.proj = nn.Linear(2 * hidden_dim, embed_dim, bias=False)
        self.head = nn.Linear(embed_dim, vocab_size, bias=True)
        self.head.weight = self.embed.weight
        
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x):
        e = self.lstm(self.drop(self.embed(x)))
        return self.head(self.proj(self.drop(self.norm(e))))

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, is_mlm=False):
    model.train(); total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x); B, T, V = logits.shape
        loss = criterion(logits.view(B*T, V), y.view(B*T))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if is_mlm:
            mask = (y != -100)
            total += loss.item() * mask.sum().item(); n += mask.sum().item()
        else:
            total += loss.item() * B * T; n += B * T
    return total / max(n, 1)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, is_mlm=False):
    model.eval(); total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x); B, T, V = logits.shape
        loss = criterion(logits.view(B*T, V), y.view(B*T))
        
        if is_mlm:
            mask = (y != -100)
            total += loss.item() * mask.sum().item(); n += mask.sum().item()
        else:
            total += loss.item() * B * T; n += B * T
    return total / max(n, 1)

# ─────────────────────────────────────────────────────────────────────────────
# QUALITATIVE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_nwp(model, vocab, device, seeds, seq_len, n_words=15, mode="greedy", temperature=0.8, top_k=10):
    model.eval()
    results =[]
    for seed in seeds:
        tokens = seed.lower().split()
        ids = vocab.encode(tokens)
        ids = ([vocab.pad_idx] * (seq_len - len(ids)) + ids
               if len(ids) < seq_len else ids[-seq_len:])
        generated = list(tokens)
        for _ in range(n_words):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(x)[:, -1, :]
            if mode == "greedy":
                pred = logits.argmax(-1).item()
            else:
                logits = logits / temperature
                top_vals, top_ids = logits.topk(top_k, dim=-1)
                probs = torch.softmax(top_vals, dim=-1)
                idx = torch.multinomial(probs, 1).item()
                pred = top_ids[0, idx].item()
            generated.append(vocab.decode([pred])[0])
            ids = ids[1:] + [pred]
        results.append(" ".join(generated))
    return results

@torch.no_grad()
def generate_mlm_samples(model, vocab, device, val_stream, seq_len, n=5):
    model.eval(); out =[]
    for i in range(n):
        ids = val_stream[i*seq_len : (i+1)*seq_len]
        if len(ids) < seq_len: break
        masked, mask_pos = list(ids),[]
        for j, tok in enumerate(ids):
            if random.random() < MASK_PROB:
                masked[j] = vocab.mask_idx; mask_pos.append(j)
        x = torch.tensor([masked], dtype=torch.long, device=device)
        preds = model(x).argmax(-1).squeeze(0).tolist()
        pred_sent = list(vocab.decode(ids))
        for pos in mask_pos: pred_sent[pos] = vocab.decode([preds[pos]])[0]
        out.append({"original": " ".join(vocab.decode(ids)),
                    "masked": " ".join(vocab.decode(masked)),
                    "predicted": " ".join(pred_sent)})
    return out

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2a — TRAIN SSM
# ─────────────────────────────────────────────────────────────────────────────
def run_ssm(vocab, train_sents, val_sents, test_sents, device, dirs, log):
    log.info("=" * 60)
    log.info("TASK 2a — S4D (Diagonal State Space), Autoregressive NWP (v11)")
    log.info("=" * 60)
    train_ds = NWPStreamDataset(train_sents, vocab, SSM_SEQ_LEN)
    val_ds = NWPStreamDataset(val_sents, vocab, SSM_SEQ_LEN)
    test_ds = NWPStreamDataset(test_sents, vocab, SSM_SEQ_LEN)
    
    train_dl = DataLoader(train_ds, batch_size=SSM_BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=SSM_BATCH, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=SSM_BATCH, shuffle=False, num_workers=0)
    
    model = SSMLanguageModel(
        len(vocab), SSM_D_MODEL, SSM_D_STATE,
        SSM_N_LAYERS, SSM_DROPOUT, vocab.pad_idx).to(device)
        
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"SSM params: {n_params:,}")
    
    optimizer = create_optimizer(model, SSM_LR, SSM_WEIGHT_DECAY)
    scheduler = get_warmup_cosine_scheduler(optimizer, SSM_WARMUP, SSM_EPOCHS)
    ls_crit = LabelSmoothingLoss(len(vocab), SSM_LABEL_SMOOTH, vocab.pad_idx)
    ce_crit = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    
    run = None
    if WANDB_AVAILABLE:
        run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name="SSM_NWP_v11", reinit="finish_previous",
            config=dict(model="S4D_Gated", task="NWP", version=11,
                        d_model=SSM_D_MODEL, d_state=SSM_D_STATE,
                        n_layers=SSM_N_LAYERS, seq_len=SSM_SEQ_LEN,
                        epochs=SSM_EPOCHS, lr=SSM_LR, warmup=SSM_WARMUP,
                        batch=SSM_BATCH, dropout=SSM_DROPOUT,
                        weight_decay=SSM_WEIGHT_DECAY, label_smooth=SSM_LABEL_SMOOTH,
                        patience=SSM_PATIENCE, vocab_size=len(vocab), n_params=n_params))
                        
    best_val, best_path = float("inf"), dirs["models"] / "ssm_nwp_best.pt"
    no_improve, stopped = 0, SSM_EPOCHS
    tr_losses, vl_losses, tr_ppx, vl_ppx = [], [], [],[]
    
    for epoch in range(1, SSM_EPOCHS + 1):
        t0 = time.time()
        train_epoch(model, train_dl, optimizer, ls_crit, device, is_mlm=False)
        tl = eval_epoch(model, train_dl, ce_crit, device, is_mlm=False)
        vl = eval_epoch(model, val_dl, ce_crit, device, is_mlm=False)
        scheduler.step()
        
        tp, vp = perplexity(tl), perplexity(vl)
        tr_losses.append(tl); vl_losses.append(vl)
        tr_ppx.append(tp); vl_ppx.append(vp)
        lr_now = optimizer.param_groups[0]["lr"]
        
        log.info(f"[SSM] {epoch:>3}/{SSM_EPOCHS} | Train loss={tl:.4f} ppl={tp:.2f} | Val loss={vl:.4f} ppl={vp:.2f} | lr={lr_now:.2e} | {time.time()-t0:.1f}s")
        if run: run.log({"ssm/train_loss": tl, "ssm/val_loss": vl, "ssm/train_ppl": tp, "ssm/val_ppl": vp, "ssm/lr": lr_now, "ssm/epoch": epoch})
        
        if vl < best_val:
            best_val = vl; no_improve = 0
            torch.save({"model_state": model.state_dict(),
                        "config": dict(vocab_size=len(vocab), d_model=SSM_D_MODEL, d_state=SSM_D_STATE, n_layers=SSM_N_LAYERS, dropout=SSM_DROPOUT, pad_idx=vocab.pad_idx)}, best_path)
            log.info(f" -> best model saved (val_loss={vl:.4f} ppl={vp:.2f})")
        else:
            no_improve += 1
            if no_improve >= SSM_PATIENCE:
                log.info(f" [Early stop] {SSM_PATIENCE} epochs no improvement.")
                stopped = epoch; break
                
    torch.save({"model_state": model.state_dict(),
                "config": dict(vocab_size=len(vocab), d_model=SSM_D_MODEL, d_state=SSM_D_STATE, n_layers=SSM_N_LAYERS, dropout=SSM_DROPOUT, pad_idx=vocab.pad_idx),
                "train_losses": tr_losses, "val_losses": vl_losses, "train_ppx": tr_ppx, "val_ppx": vl_ppx, "stopped_epoch": stopped}, dirs["models"] / "ssm_nwp.pt")
                
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    test_loss = eval_epoch(model, test_dl, ce_crit, device, is_mlm=False)
    test_ppl = perplexity(test_loss)
    log.info(f"[SSM] TEST -> loss={test_loss:.4f} ppl={test_ppl:.2f}")
    
    if run: run.log({"ssm/test_loss": test_loss, "ssm/test_ppl": test_ppl})
    plot_curves(tr_losses, vl_losses, "Loss", "S4D — NWP Loss (v11)", dirs["plots"] / "ssm_loss.png")
    plot_curves(tr_ppx, vl_ppx, "Perplexity", "S4D — NWP Perplexity (v11)", dirs["plots"] / "ssm_perplexity.png")
    
    seeds =["the jury said", "the city of", "fulton county grand", "it recommended that", "the election was", "the president of the", "police department the jury"]
    gens_greedy = generate_nwp(model, vocab, device, seeds, SSM_SEQ_LEN, n_words=15, mode="greedy")
    random.seed(SEED)
    gens_topk = generate_nwp(model, vocab, device, seeds, SSM_SEQ_LEN, n_words=15, mode="topk", temperature=0.8, top_k=10)
    
    @torch.no_grad()
    def score_next_words(model, vocab, device, context, seq_len):
        tokens = context.lower().split()
        ids = vocab.encode(tokens)
        ids = ([vocab.pad_idx]*(seq_len-len(ids))+ids if len(ids)<seq_len else ids[-seq_len:])
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)[:, -1, :] 
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        top5_vals, top5_ids = probs.topk(5)
        return [(vocab.decode([i.item()])[0], f"{v.item()*100:.2f}%") for v, i in zip(top5_vals, top5_ids)]
        
    demo_contexts =["the fulton county grand jury said", "the city of atlanta for the", "it recommended that fulton legislators"]
    lines =["=" * 70, "S4D — Next-Word Prediction Results (v11)", "=" * 70,
             f"Vocab size : {len(vocab)}", f"Parameters : {n_params:,}",
             f"Architecture : S4D_Gated, d_model={SSM_D_MODEL}, d_state={SSM_D_STATE}",
             f"Best val loss : {best_val:.4f} (ppl={perplexity(best_val):.2f})",
             f"Test loss : {test_loss:.4f} (ppl={test_ppl:.2f})", "",
             "─" * 70, "EXAMPLE 1: Greedy Continuations (+15 words)", "─" * 70]
    for seed, gen in zip(seeds, gens_greedy):
        lines +=[f"Seed : {seed}", f"Output: {gen}", ""]
    lines +=["─" * 70, "EXAMPLE 2: Top-K Sampled Continuations (k=10, temp=0.8)", "─" * 70]
    for seed, gen in zip(seeds, gens_topk):
        lines +=[f"Seed : {seed}", f"Output: {gen}", ""]
    lines +=["─" * 70, "EXAMPLE 3: Top-5 Most Likely Next Words (word scoring)", "─" * 70]
    for ctx in demo_contexts:
        top5 = score_next_words(model, vocab, device, ctx, SSM_SEQ_LEN)
        lines +=[f"Context : '{ctx}'", f"Top-5 : {', '.join(f'{w}({p})' for w, p in top5)}", ""]
        
    (dirs["results"] / "task2_ssm.txt").write_text("\n".join(lines))
    if run: run.finish()
    return model, {"train_loss": tr_losses, "val_loss": vl_losses, "train_ppl": tr_ppx, "val_ppl": vl_ppx, "test_loss": test_loss, "test_ppl": test_ppl}

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2b — TRAIN Bi-LSTM
# ─────────────────────────────────────────────────────────────────────────────
def run_bilstm(vocab, train_sents, val_sents, test_sents, device, dirs, log):
    log.info("=" * 60)
    log.info("TASK 2b — Bi-LSTM MLM (v11, Custom Recurrence)")
    log.info("=" * 60)
    train_ds = MLMStreamDataset(train_sents, vocab, BILSTM_SEQ_LEN, BILSTM_STRIDE, MASK_PROB)
    val_ds = MLMStreamDataset(val_sents, vocab, BILSTM_SEQ_LEN, BILSTM_STRIDE, MASK_PROB)
    test_ds = MLMStreamDataset(test_sents, vocab, BILSTM_SEQ_LEN, BILSTM_STRIDE, MASK_PROB)
    val_stream = vocab.encode(sentences_to_stream(val_sents, vocab.EOS))
    
    train_dl = DataLoader(train_ds, batch_size=BILSTM_BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BILSTM_BATCH, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=BILSTM_BATCH, shuffle=False, num_workers=0)
    
    model = BiLSTMMLM(
        len(vocab), BILSTM_EMBED, BILSTM_HIDDEN,
        BILSTM_LAYERS, BILSTM_DROPOUT, vocab.pad_idx).to(device)
        
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Bi-LSTM params: {n_params:,}")
    
    optimizer = create_optimizer(model, BILSTM_LR, BILSTM_WEIGHT_DECAY)
    scheduler = get_warmup_cosine_scheduler(optimizer, BILSTM_WARMUP, BILSTM_EPOCHS)
    ls_crit = LabelSmoothingLoss(len(vocab), BILSTM_LABEL_SMOOTH, -100)
    ce_crit = nn.CrossEntropyLoss(ignore_index=-100)
    
    run = None
    if WANDB_AVAILABLE:
        run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name="BiLSTM_MLM_v11", reinit="finish_previous",
            config=dict(model="CustomBiLSTM", task="MLM", version=11,
                        embed_dim=BILSTM_EMBED, hidden_dim=BILSTM_HIDDEN,
                        n_layers=BILSTM_LAYERS, seq_len=BILSTM_SEQ_LEN, stride=BILSTM_STRIDE,
                        epochs=BILSTM_EPOCHS, lr=BILSTM_LR, warmup=BILSTM_WARMUP, batch=BILSTM_BATCH,
                        dropout=BILSTM_DROPOUT, weight_decay=BILSTM_WEIGHT_DECAY,
                        label_smooth=BILSTM_LABEL_SMOOTH, patience=BILSTM_PATIENCE,
                        mask_prob=MASK_PROB, vocab_size=len(vocab), n_params=n_params))
                        
    best_val, best_path = float("inf"), dirs["models"] / "bilstm_mlm_best.pt"
    no_improve, stopped = 0, BILSTM_EPOCHS
    tr_losses, vl_losses, tr_ppx, vl_ppx = [], [], [],[]
    
    for epoch in range(1, BILSTM_EPOCHS + 1):
        t0 = time.time()
        train_epoch(model, train_dl, optimizer, ls_crit, device, is_mlm=True)
        tl = eval_epoch(model, train_dl, ce_crit, device, is_mlm=True)
        vl = eval_epoch(model, val_dl, ce_crit, device, is_mlm=True)
        scheduler.step()
        
        tp, vp = perplexity(tl), perplexity(vl)
        tr_losses.append(tl); vl_losses.append(vl)
        tr_ppx.append(tp); vl_ppx.append(vp)
        lr_now = optimizer.param_groups[0]["lr"]
        
        log.info(f"[BiLSTM] {epoch:>3}/{BILSTM_EPOCHS} | Train loss={tl:.4f} ppl={tp:.2f} | Val loss={vl:.4f} ppl={vp:.2f} | lr={lr_now:.2e} | {time.time()-t0:.1f}s")
        if run: run.log({"bilstm/train_loss": tl, "bilstm/val_loss": vl, "bilstm/train_ppl": tp, "bilstm/val_ppl": vp, "bilstm/lr": lr_now, "bilstm/epoch": epoch})
        
        if vl < best_val:
            best_val = vl; no_improve = 0
            torch.save({"model_state": model.state_dict(),
                        "config": dict(vocab_size=len(vocab), embed_dim=BILSTM_EMBED, hidden_dim=BILSTM_HIDDEN, n_layers=BILSTM_LAYERS, dropout=BILSTM_DROPOUT, pad_idx=vocab.pad_idx)}, best_path)
            log.info(f" -> best model saved (val_loss={vl:.4f} ppl={vp:.2f})")
        else:
            no_improve += 1
            if no_improve >= BILSTM_PATIENCE:
                log.info(f" [Early stop] {BILSTM_PATIENCE} epochs no improvement.")
                stopped = epoch; break
                
    torch.save({"model_state": model.state_dict(), "config": dict(vocab_size=len(vocab), embed_dim=BILSTM_EMBED, hidden_dim=BILSTM_HIDDEN, n_layers=BILSTM_LAYERS, dropout=BILSTM_DROPOUT, pad_idx=vocab.pad_idx),
                "train_losses": tr_losses, "val_losses": vl_losses, "train_ppx": tr_ppx, "val_ppx": vl_ppx, "stopped_epoch": stopped}, dirs["models"] / "bilstm_mlm.pt")
                
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    test_loss = eval_epoch(model, test_dl, ce_crit, device, is_mlm=True)
    test_ppl = perplexity(test_loss)
    log.info(f"[BiLSTM] TEST -> loss={test_loss:.4f} ppl={test_ppl:.2f}")
    
    if run: run.log({"bilstm/test_loss": test_loss, "bilstm/test_ppl": test_ppl})
    plot_curves(tr_losses, vl_losses, "Loss", "Bi-LSTM — MLM Loss (v11)", dirs["plots"] / "bilstm_loss.png")
    plot_curves(tr_ppx, vl_ppx, "Perplexity", "Bi-LSTM — MLM Perplexity (v11)", dirs["plots"] / "bilstm_perplexity.png")
    
    samples = generate_mlm_samples(model, vocab, device, val_stream, BILSTM_SEQ_LEN, n=5)
    
    @torch.no_grad()
    def fill_single_mask(model, vocab, device, sentence_with_mask, seq_len):
        tokens = sentence_with_mask.lower().split()
        ids = vocab.encode(tokens)
        ids = (ids + [vocab.pad_idx]*seq_len)[:seq_len]
        mask_positions = [i for i, t in enumerate(tokens) if t == "<mask>"]
        if not mask_positions: return[]
        for pos in mask_positions:
            if pos < len(ids): ids[pos] = vocab.mask_idx
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x).squeeze(0) 
        results =[]
        for pos in mask_positions:
            if pos >= logits.size(0): continue
            probs = torch.softmax(logits[pos], dim=-1)
            top5_v, top5_i = probs.topk(5)
            top5 = [(vocab.decode([i.item()])[0], f"{v.item()*100:.1f}%") for v, i in zip(top5_v, top5_i)]
            results.append((pos, top5))
        return results

    demo_sentences =[
        "the fulton county grand jury said <mask> an investigation",
        "the city of <mask> for the manner in which the election",
        "it recommended that fulton <mask> act to have these laws",
        "the jury said it did find that many of <mask> registration",
        "merger <mask> however the jury said it believes",
    ]
    
    lines =["=" * 70, "Bi-LSTM — Masked Language Modeling Results (v11)", "=" * 70,
             f"Vocab size : {len(vocab)}", f"Parameters : {n_params:,}",
             f"Architecture : Custom BiLSTM, hidden={BILSTM_HIDDEN}, layers={BILSTM_LAYERS}",
             f"Best val loss : {best_val:.4f} (ppl={perplexity(best_val):.2f})",
             f"Test loss : {test_loss:.4f} (ppl={test_ppl:.2f})", "",
             "─" * 70, "EXAMPLE 1: Random Masking (15% of tokens) — Fill-in-the-Blank", "─" * 70]
    for i, s in enumerate(samples, 1):
        lines += [f"Sample {i}", f" Original : {s['original']}", f" Masked : {s['masked']}", f" Predicted: {s['predicted']}", ""]
    lines +=["─" * 70, "EXAMPLE 2: Targeted Single-Word Fill (shows bidirectional context)", "─" * 70]
    for sent in demo_sentences:
        results = fill_single_mask(model, vocab, device, sent, BILSTM_SEQ_LEN)
        lines +=[f"Sentence : {sent}"]
        for pos, top5 in results: lines +=[f" Mask at pos {pos}: {', '.join(f'{w}({p})' for w, p in top5)}"]
        lines += [""]
        
    (dirs["results"] / "task2_bilstm.txt").write_text("\n".join(lines))
    if run: run.finish()
    return model, {"train_loss": tr_losses, "val_loss": vl_losses, "train_ppl": tr_ppx, "val_ppl": vl_ppx, "test_loss": test_loss, "test_ppl": test_ppl}

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def write_comparison(sm, bm, dirs, log):
    lines =[
        "=" * 70, "TASK 2 — Model Comparison (v11)", "=" * 70,
        f"{'Metric':<30} {'S4D (NWP)':>15} {'Bi-LSTM (MLM)':>15}", "-" * 62,
        f"{'Best Val Loss':<30} {min(sm['val_loss']):>15.4f} {min(bm['val_loss']):>15.4f}",
        f"{'Best Val Perplexity':<30} {min(sm['val_ppl']):>15.2f} {min(bm['val_ppl']):>15.2f}",
        f"{'Test Loss':<30} {sm['test_loss']:>15.4f} {bm['test_loss']:>15.4f}",
        f"{'Test Perplexity':<30} {sm['test_ppl']:>15.2f} {bm['test_ppl']:>15.2f}",
        "",
        "Notes:",
        " S4D : Causal autoregressive NWP using FFT Convolution over frozen HiPPO/LegS complex state.",
        " Bi-LSTM: Custom bidirectional MLM, hidden=256, seq_len=128.",
    ]
    (dirs["results"] / "task2_comparison.txt").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models =["S4D (NWP)", "Bi-LSTM (MLM)"]; colors =["#4C72B0", "#DD8452"]
    for ax, vals, title in zip(axes, [[min(sm["val_ppl"]), min(bm["val_ppl"])], [sm["test_ppl"], bm["test_ppl"]]], ["Best Validation Perplexity", "Test Perplexity"]):
        bars = ax.bar(models, vals, color=colors, width=0.5)
        ax.set_title(title, fontsize=12); ax.set_ylabel("Perplexity")
        for bar, v in zip(bars, vals): ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")
    plt.suptitle("Task 2 — S4D vs Bi-LSTM (v11)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "task2_comparison.png", dpi=150); plt.close(fig)
    log.info("Comparison summary saved.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    set_seed(SEED)
    device = get_device()
    print(f"[INFO] Device: {device}")
    
    dirs = setup_dirs()
    log = setup_logging(dirs["logs"] / "task2.log")
    log.info(f"Data: {DATA_PATH} | Output: {OUTPUT_DIR} | Device: {device}")
    
    sentences = load_and_tokenize(DATA_PATH)
    train_sents, val_sents, test_sents = split_sentences(sentences)
    log.info(f"Split -> train={len(train_sents)} val={len(val_sents)} test={len(test_sents)}")
    
    train_stream = sentences_to_stream(train_sents, "<EOS>")
    vocab = Vocabulary(min_freq=3)
    vocab.build(train_stream)
    
    (dirs["models"] / "vocab.json").write_text(json.dumps({
        "word2idx": vocab.word2idx,
        "idx2word": {str(k): v for k, v in vocab.idx2word.items()}}, indent=2))
        
    _, ssm_metrics = run_ssm( vocab, train_sents, val_sents, test_sents, device, dirs, log)
    _, bilstm_metrics = run_bilstm(vocab, train_sents, val_sents, test_sents, device, dirs, log)
    write_comparison(ssm_metrics, bilstm_metrics, dirs, log)
    
    log.info("=" * 60)
    log.info(f"Task 2 complete. Outputs in: {OUTPUT_DIR}/")
    log.info("=" * 60)