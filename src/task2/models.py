import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
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

class S4D(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.h = d_model
        self.n = d_state

        log_dt = torch.rand(self.h) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        self.log_dt = nn.Parameter(log_dt)

        A_real = torch.full((self.h, self.n), -0.5)
        A_imag = math.pi * (torch.arange(self.n) + 0.5).unsqueeze(0).repeat(self.h, 1)
        
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