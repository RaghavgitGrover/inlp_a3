import math
import random
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def tokenize_line(plain_line: str, cipher_line: str) -> Optional[List[str]]:
    """SPACE -> 1 cipher digit, NON-SPACE -> 2 cipher digits."""
    sp = plain_line.count(' '); ns = len(plain_line) - sp
    if sp + ns * 2 != len(cipher_line):
        return None
    tokens, c =[], 0
    for pc in plain_line:
        if pc == ' ':
            tokens.append(cipher_line[c]); c += 1
        else:
            tokens.append(cipher_line[c:c+2]); c += 2
    return tokens

def build_samples(plain_lines, cipher_lines, max_line_len):
    samples =[]
    for pl, cl in zip(plain_lines, cipher_lines):
        if len(pl) == 0 or len(pl) > max_line_len:
            continue
        tokens = tokenize_line(pl, cl)
        if tokens is None:
            continue
        samples.append((tokens, pl))
    return samples

def build_token_vocab(seqs):
    unique = set(t for s in seqs for t in s)
    t2i = {"<PAD>":0, "<UNK>":1}
    for t in sorted(unique):
        if t not in t2i: t2i[t] = len(t2i)
    return t2i, {v:k for k,v in t2i.items()}

def build_char_vocab(texts):
    unique = set(c for t in texts for c in t)
    c2i = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
    for c in sorted(unique):
        if c not in c2i: c2i[c] = len(c2i)
    return c2i, {v:k for k,v in c2i.items()}

class BucketSampler(Sampler):
    def __init__(self, lengths: List[int], batch_size: int,
                 bucket_width: int = 10, shuffle: bool = True):
        self.lengths      = lengths
        self.batch_size   = batch_size
        self.bucket_width = bucket_width
        self.shuffle      = shuffle

    def __iter__(self):
        buckets: Dict[int, List[int]] = {}
        for i, l in enumerate(self.lengths):
            key = (l // self.bucket_width) * self.bucket_width
            buckets.setdefault(key,[]).append(i)

        all_batches =[]
        for key in buckets:
            indices = buckets[key]
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                all_batches.append(indices[start:start+self.batch_size])

        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield from batch

    def __len__(self):
        return len(self.lengths)

class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, bucket_width=10, shuffle=True):
        self.lengths      = lengths
        self.batch_size   = batch_size
        self.bucket_width = bucket_width
        self.shuffle      = shuffle

    def __iter__(self):
        buckets: Dict[int, List[int]] = {}
        for i, l in enumerate(self.lengths):
            key = (l // self.bucket_width) * self.bucket_width
            buckets.setdefault(key,[]).append(i)

        all_batches =[]
        for key in buckets:
            indices = buckets[key][:]
            if self.shuffle: random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                b = indices[start:start+self.batch_size]
                if b: all_batches.append(b)

        if self.shuffle: random.shuffle(all_batches)
        yield from all_batches

    def __len__(self):
        return sum(
            math.ceil(len(v) / self.batch_size)
            for v in self._buckets().values()
        )

    def _buckets(self):
        b = {}
        for i, l in enumerate(self.lengths):
            b.setdefault((l//self.bucket_width)*self.bucket_width,[]).append(i)
        return b

class CipherDataset(Dataset):
    def __init__(self, samples, t2i, c2i):
        self.samples = samples
        self.t2i = t2i; self.c2i = c2i

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        tokens, text = self.samples[i]
        src =[self.t2i.get(t, 1) for t in tokens]
        tgt = [self.c2i["<SOS>"]] + [self.c2i.get(c, 3) for c in text] + [self.c2i["<EOS>"]]
        return src, tgt, len(tokens)

def pad_collate(batch):
    srcs, tgts, _ = zip(*batch)
    max_s = max(len(s) for s in srcs)
    max_t = max(len(t) for t in tgts)
    src_pad = torch.zeros(len(srcs), max_s, dtype=torch.long)
    tgt_pad = torch.zeros(len(tgts), max_t, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, :len(s)] = torch.tensor(s)
        tgt_pad[i, :len(t)] = torch.tensor(t)
    return src_pad, tgt_pad

def make_loader(samples, t2i, c2i, batch_size, bucket_width, shuffle):
    ds = CipherDataset(samples, t2i, c2i)
    lengths = [len(s[0]) for s in samples]
    batch_sampler = BucketBatchSampler(lengths, batch_size, bucket_width, shuffle)
    return DataLoader(ds, batch_sampler=batch_sampler, collate_fn=pad_collate, num_workers=2, pin_memory=True)