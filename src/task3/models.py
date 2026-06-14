import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# TASK-1 MODEL DEFINITIONS (Decryption)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMCell(nn.Module):
    def __init__(self, in_size: int, H: int):
        super().__init__()
        self.H   = H
        self.W_ih = nn.Linear(in_size, 4 * H)
        self.W_hh = nn.Linear(H, 4 * H, bias=False)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = (self.W_ih(x) + self.W_hh(h)).chunk(4, dim=-1)
        c2 = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c2), c2

    def init_s(self, B, dev):
        z = torch.zeros(B, self.H, device=dev)
        return z, z.clone()

class BahdanauAttention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, align_dim: int = 256):
        super().__init__()
        self.Wq = nn.Linear(query_dim, align_dim, bias=False)
        self.Wk = nn.Linear(key_dim,   align_dim, bias=False)
        self.v  = nn.Linear(align_dim, 1,          bias=False)

    def forward(self, q, K, mask=None):
        e = self.v(torch.tanh(self.Wq(q).unsqueeze(1) + self.Wk(K))).squeeze(-1)
        if mask is not None:
            e = e.masked_fill(mask, float("-inf"))
        a = F.softmax(e, dim=-1)
        return torch.bmm(a.unsqueeze(1), K).squeeze(1), a

class LSTMEncoder(nn.Module):
    def __init__(self, vocab: int, E: int, H: int, L: int, D: float):
        super().__init__()
        self.H, self.L = H, L
        self.embed = nn.Embedding(vocab, E, padding_idx=0)
        self.drop  = nn.Dropout(D)
        self.cells = nn.ModuleList([LSTMCell(E if i == 0 else H, H) for i in range(L)])
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])

    def forward(self, src):
        B, T  = src.shape; 
        dev = src.device
        pad_mask = (src == 0)
        x = self.drop(self.embed(src))
        last_c = None
        for l in range(self.L):
            fc, g_inp = self.cells[l], self.cells[l].W_ih(x)
            h, c = fc.init_s(B, dev)
            fwd =[]
            for t in range(T):
                gates = g_inp[:, t] + fc.W_hh(h)
                i, f, g, o = gates.chunk(4, dim=-1)
                c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
                h = torch.sigmoid(o) * torch.tanh(c)
                fwd.append(h)
            x = self.drop(self.norms[l](torch.stack(fwd, 1)))
            if l == self.L - 1: last_c = c
        return x, (x[:, -1, :], last_c), pad_mask

class LSTMDecoder(nn.Module):
    def __init__(self, vocab: int, E: int, H: int, D: float):
        super().__init__()
        self.embed  = nn.Embedding(vocab, E, padding_idx=0)
        self.drop   = nn.Dropout(D)
        self.attn   = BahdanauAttention(H, H)
        self.cell   = LSTMCell(E + H, H)
        self.fc_out = nn.Linear(H, vocab)

    def step(self, tok, state, enc_out, mask):
        h, c = state
        e = self.drop(self.embed(tok))
        ctx, a = self.attn(h, enc_out, mask)
        h, c = self.cell(torch.cat([e, ctx], -1), (h, c))
        return self.fc_out(h), (h, c), a

class Seq2SeqLSTM(nn.Module):
    def __init__(self, enc: LSTMEncoder, dec: LSTMDecoder):
        super().__init__()
        self.encoder = enc
        self.decoder = dec

    @torch.no_grad()
    def greedy_with_conf(self, src, c2i: dict, i2c: dict, max_len: int):
        self.eval()
        enc_out, state, mask = self.encoder(src)
        B, SOS, EOS = src.size(0), c2i["<SOS>"], c2i["<EOS>"]
        tok = torch.full((B,), SOS, dtype=torch.long, device=src.device)
        seqs, confs, done = [[] for _ in range(B)], [[] for _ in range(B)], [False] * B
        for _ in range(max_len):
            logits, state, _ = self.decoder.step(tok, state, enc_out, mask)
            probs, tok = F.softmax(logits, dim=-1), logits.argmax(-1)
            for b in range(B):
                if not done[b]:
                    idx, conf = tok[b].item(), probs[b, tok[b]].item()
                    if idx == EOS: done[b] = True
                    else: seqs[b].append(i2c.get(idx, "")); confs[b].append(conf)
            if all(done): break
        return["".join(s) for s in seqs], confs

# ─────────────────────────────────────────────────────────────────────────────
# TASK-2 MODEL DEFINITIONS (SSM / Bi-LSTM)
# ─────────────────────────────────────────────────────────────────────────────
class S4D(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.h, self.n = d_model, d_state
        log_dt = (torch.rand(self.h) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.log_dt = nn.Parameter(log_dt)
        A_real = torch.full((self.h, self.n), -0.5)
        A_imag = (math.pi * (torch.arange(self.n) + 0.5).unsqueeze(0).repeat(self.h, 1))
        self.log_A_real = nn.Parameter(torch.log(torch.abs(A_real)), requires_grad=False)
        self.A_imag = nn.Parameter(A_imag, requires_grad=False)
        self.B = nn.Parameter(torch.randn(self.h, self.n, dtype=torch.cfloat) * 0.1)
        self.C = nn.Parameter(torch.randn(self.h, self.n, dtype=torch.cfloat) * 0.1)
        self.D = nn.Parameter(torch.randn(self.h) * 0.1)

    def forward(self, x):
        b, l, h = x.shape; 
        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag; 
        dtA = A * dt.unsqueeze(-1)
        B_d = (torch.exp(dtA) - 1.0) / A * self.B; 
        t = torch.arange(l, device=x.device)
        K = torch.einsum('hn, hnl -> hl', self.C * B_d, torch.exp(dtA.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0))).real
        x_t = x.transpose(1, 2)
        K_f = torch.fft.rfft(K, n=2 * l); 
        x_f = torch.fft.rfft(x_t, n=2 * l)
        y = torch.fft.irfft(x_f * K_f, n=2 * l)[..., :l]
        return y.transpose(1, 2) + x * self.D

class GatedSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.expand = nn.Linear(d_model, 2 * d_model, bias=False)
        self.ssm = S4D(d_model, d_state)
        self.contract = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(2 * d_model, d_model), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_n = self.norm1(x); 
        v, gate = self.expand(x_n).chunk(2, dim=-1)
        x = x + self.dropout(self.contract(self.ssm(v) * F.silu(gate)))
        return x + self.ffn(self.norm2(x))

class SSMLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, n_layers, dropout, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.edrop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([GatedSSMBlock(d_model, d_state, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=True)
        self.head.weight = self.embed.weight; 
        nn.init.normal_(self.embed.weight, std=0.02); 
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        e = self.edrop(self.embed(x))
        for blk in self.blocks: e = blk(e)
        return self.head(self.norm(e))

class _CustomLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = False):
        super().__init__(); 
        self.hidden_size = hidden_size; 
        self.bidirectional = bidirectional
        self.W_ih_f = nn.Linear(input_size, 4 * hidden_size); 
        self.W_hh_f = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        if bidirectional: self.W_ih_b = nn.Linear(input_size, 4 * hidden_size); self.W_hh_b = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def _lstm_dir(self, x, W_ih, W_hh, reverse=False):
        B, T, _ = x.shape; dev = x.device; 
        gx = W_ih(x)
        h = torch.zeros(B, self.hidden_size, device=dev); 
        c = torch.zeros(B, self.hidden_size, device=dev)
        out =[]; 
        rng = range(T - 1, -1, -1) if reverse else range(T)
        for t in rng:
            gates = gx[:, t] + W_hh(h); 
            i, f, g, o = gates.chunk(4, dim=-1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g); 
            h = torch.sigmoid(o) * torch.tanh(c)
            out.append(h)
        if reverse: out = out[::-1]
        return torch.stack(out, dim=1)

    def forward(self, x):
        fwd = self._lstm_dir(x, self.W_ih_f, self.W_hh_f, reverse=False)
        if self.bidirectional: return torch.cat([fwd, self._lstm_dir(x, self.W_ih_b, self.W_hh_b, reverse=True)], dim=-1)
        return fwd

class _MultiLayerBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__(); 
        self.layers = nn.ModuleList(); 
        self.dropouts = nn.ModuleList(); 
        self.num_layers = num_layers
        for i in range(num_layers):
            self.layers.append(_CustomLSTM(input_size if i == 0 else hidden_size * 2, hidden_size, bidirectional=True))
            if i < num_layers - 1: self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1: x = self.dropouts[i](x)
        return x

class BiLSTMMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = _MultiLayerBiLSTM(embed_dim, hidden_dim, n_layers, dropout); self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(2 * hidden_dim); 
        self.proj = nn.Linear(2 * hidden_dim, embed_dim, bias=False)
        self.head = nn.Linear(embed_dim, vocab_size, bias=True)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02); 
        nn.init.zeros_(self.head.bias); 
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x): 
        return self.head(self.proj(self.drop(self.norm(self.lstm(self.drop(self.embed(x)))))))