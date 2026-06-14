import random
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNCell(nn.Module):
    def __init__(self, in_size: int, H: int):
        super().__init__()
        self.H = H
        self.W_ih = nn.Linear(in_size, H)
        self.W_hh = nn.Linear(H, H, bias=False)

    def forward(self, x, h):
        return torch.tanh(self.W_ih(x) + self.W_hh(h))

    def init_h(self, B, dev):
        return torch.zeros(B, self.H, device=dev)

class LSTMCell(nn.Module):
    def __init__(self, in_size: int, H: int):
        super().__init__()
        self.H = H
        self.W_ih = nn.Linear(in_size, 4*H)
        self.W_hh = nn.Linear(H, 4*H, bias=False)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = (self.W_ih(x) + self.W_hh(h)).chunk(4, dim=-1)
        c2 = torch.sigmoid(f)*c + torch.sigmoid(i)*torch.tanh(g)
        return torch.sigmoid(o)*torch.tanh(c2), c2

    def init_s(self, B, dev):
        z = torch.zeros(B, self.H, device=dev); return z, z.clone()

class BahdanauAttention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, align_dim: int = 256):
        super().__init__()
        self.Wq = nn.Linear(query_dim, align_dim, bias=False)
        self.Wk = nn.Linear(key_dim,   align_dim, bias=False)
        self.v  = nn.Linear(align_dim, 1,         bias=False)

    def forward(self, q, K, mask=None):
        e = self.v(torch.tanh(self.Wq(q).unsqueeze(1) + self.Wk(K))).squeeze(-1)
        if mask is not None:
            e = e.masked_fill(mask, float("-inf"))
        a = F.softmax(e, dim=-1)
        return torch.bmm(a.unsqueeze(1), K).squeeze(1), a

class RNNEncoder(nn.Module):
    def __init__(self, vocab, E, H, L, D):
        super().__init__()
        self.H = H; self.L = L
        self.embed = nn.Embedding(vocab, E, padding_idx=0)
        self.drop  = nn.Dropout(D)
        self.cells = nn.ModuleList([RNNCell(E if i==0 else H, H) for i in range(L)])
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])

    def forward(self, src):
        B, T = src.shape; dev = src.device
        pad_mask = (src == 0)
        x = self.drop(self.embed(src))
        for l in range(self.L):
            fc = self.cells[l]
            inp = fc.W_ih(x)
            h = fc.init_h(B, dev); fwd =[]
            for t in range(T):
                h = torch.tanh(inp[:, t] + fc.W_hh(h))
                fwd.append(h)
            x = self.drop(self.norms[l](torch.stack(fwd, 1)))
        return x, x[:, -1, :], pad_mask

class LSTMEncoder(nn.Module):
    def __init__(self, vocab, E, H, L, D):
        super().__init__()
        self.H = H; self.L = L
        self.embed = nn.Embedding(vocab, E, padding_idx=0)
        self.drop  = nn.Dropout(D)
        self.cells = nn.ModuleList([LSTMCell(E if i==0 else H, H) for i in range(L)])
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])

    def forward(self, src):
        B, T = src.shape; dev = src.device
        pad_mask = (src == 0)
        x = self.drop(self.embed(src))
        last_c = None
        for l in range(self.L):
            fc = self.cells[l]
            g_inp = fc.W_ih(x)
            h, c = fc.init_s(B, dev); fwd =[]
            for t in range(T):
                gates = g_inp[:, t] + fc.W_hh(h)
                i, f, g, o = gates.chunk(4, dim=-1)
                c = torch.sigmoid(f)*c + torch.sigmoid(i)*torch.tanh(g)
                h = torch.sigmoid(o)*torch.tanh(c)
                fwd.append(h)
            x = self.drop(self.norms[l](torch.stack(fwd, 1)))
            if l == self.L - 1:
                last_c = c
        return x, (x[:, -1, :], last_c), pad_mask

class RNNDecoder(nn.Module):
    def __init__(self, vocab, E, H, D):
        super().__init__()
        self.embed  = nn.Embedding(vocab, E, padding_idx=0)
        self.drop   = nn.Dropout(D)
        self.attn   = BahdanauAttention(query_dim=H, key_dim=H)
        self.cell   = RNNCell(E + H, H)
        self.fc_out = nn.Linear(H, vocab)

    def step(self, tok, h, enc_out, mask):
        e       = self.drop(self.embed(tok))
        ctx, a  = self.attn(h, enc_out, mask)
        h       = self.cell(torch.cat([e, ctx], -1), h)
        return self.fc_out(h), h, a

    def forward(self, tgt, h, enc_out, mask, tf=0.5):
        B, T = tgt.shape; V = self.fc_out.out_features
        out = torch.zeros(B, T, V, device=tgt.device)
        tok = tgt[:, 0]
        for t in range(1, T):
            logits, h, _ = self.step(tok, h, enc_out, mask)
            out[:, t]    = logits
            tok = tgt[:, t] if random.random() < tf else logits.argmax(-1)
        return out

class LSTMDecoder(nn.Module):
    def __init__(self, vocab, E, H, D):
        super().__init__()
        self.embed  = nn.Embedding(vocab, E, padding_idx=0)
        self.drop   = nn.Dropout(D)
        self.attn   = BahdanauAttention(query_dim=H, key_dim=H)
        self.cell   = LSTMCell(E + H, H)
        self.fc_out = nn.Linear(H, vocab)

    def step(self, tok, state, enc_out, mask):
        h, c    = state
        e       = self.drop(self.embed(tok))
        ctx, a  = self.attn(h, enc_out, mask)
        h, c    = self.cell(torch.cat([e, ctx], -1), (h, c))
        return self.fc_out(h), (h, c), a

    def forward(self, tgt, state, enc_out, mask, tf=0.5):
        B, T = tgt.shape; V = self.fc_out.out_features
        out = torch.zeros(B, T, V, device=tgt.device)
        tok = tgt[:, 0]
        for t in range(1, T):
            logits, state, _ = self.step(tok, state, enc_out, mask)
            out[:, t]        = logits
            tok = tgt[:, t] if random.random() < tf else logits.argmax(-1)
        return out

class Seq2SeqRNN(nn.Module):
    def __init__(self, enc, dec):
        super().__init__(); self.encoder=enc; self.decoder=dec

    def forward(self, src, tgt, tf=0.5):
        enc_out, h, mask = self.encoder(src)
        return self.decoder(tgt, h, enc_out, mask, tf)

    @torch.no_grad()
    def greedy(self, src, c2i, i2c, max_len):
        self.eval()
        enc_out, h, mask = self.encoder(src)
        B = src.size(0); SOS, EOS = c2i["<SOS>"], c2i["<EOS>"]
        tok = torch.full((B,), SOS, dtype=torch.long, device=src.device)
        seqs = [[] for _ in range(B)]; done = [False]*B
        for _ in range(max_len):
            logits, h, _ = self.decoder.step(tok, h, enc_out, mask)
            tok = logits.argmax(-1)
            for b in range(B):
                if not done[b]:
                    idx = tok[b].item()
                    if idx == EOS: done[b] = True
                    else: seqs[b].append(i2c.get(idx, ""))
            if all(done): break
        return ["".join(s) for s in seqs]

class Seq2SeqLSTM(nn.Module):
    def __init__(self, enc, dec):
        super().__init__(); self.encoder=enc; self.decoder=dec

    def forward(self, src, tgt, tf=0.5):
        enc_out, state, mask = self.encoder(src)
        return self.decoder(tgt, state, enc_out, mask, tf)

    @torch.no_grad()
    def greedy(self, src, c2i, i2c, max_len):
        self.eval()
        enc_out, state, mask = self.encoder(src)
        B = src.size(0); SOS, EOS = c2i["<SOS>"], c2i["<EOS>"]
        tok = torch.full((B,), SOS, dtype=torch.long, device=src.device)
        seqs = [[] for _ in range(B)]; done = [False]*B
        for _ in range(max_len):
            logits, state, _ = self.decoder.step(tok, state, enc_out, mask)
            tok = logits.argmax(-1)
            for b in range(B):
                if not done[b]:
                    idx = tok[b].item()
                    if idx == EOS: done[b] = True
                    else: seqs[b].append(i2c.get(idx, ""))
            if all(done): break
        return["".join(s) for s in seqs]

def make_model(kind, t_vocab, c_vocab, cfg):
    E, H, L, D = cfg["embed_dim"], cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]
    if kind == "rnn":
        return Seq2SeqRNN(RNNEncoder(t_vocab,E,H,L,D), RNNDecoder(c_vocab,E,H,D)).to(DEVICE)
    return Seq2SeqLSTM(LSTMEncoder(t_vocab,E,H,L,D), LSTMDecoder(c_vocab,E,H,D)).to(DEVICE)