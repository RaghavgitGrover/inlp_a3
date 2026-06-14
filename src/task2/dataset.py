import random
import torch
from typing import List, Dict
from torch.utils.data import Dataset

class Vocabulary:
    PAD, UNK, MASK, EOS = "<PAD>", "<UNK>", "<MASK>", "<EOS>"
    def __init__(self, min_freq=3):
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
        return [self.word2idx.get(t, unk) for t in tokens]

    def decode(self, ids):
        return[self.idx2word.get(i, self.UNK) for i in ids]

    def __len__(self): return len(self.word2idx)
    @property
    def pad_idx(self): return self.word2idx[self.PAD]
    @property
    def mask_idx(self): return self.word2idx[self.MASK]
    @property
    def eos_idx(self): return self.word2idx[self.EOS]

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

def split_sentences(sentences, train_split, val_split):
    random.shuffle(sentences)
    n = len(sentences)
    t = int(n * train_split)
    v = int(n * (train_split + val_split))
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