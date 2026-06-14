import os
import sys
import math
import json
import time
import random
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from src.task2.dataset import (
    Vocabulary, load_and_tokenize, sentences_to_stream, 
    split_sentences, NWPStreamDataset, MLMStreamDataset
)
from src.task2.models import SSMLanguageModel, BiLSTMMLM, LabelSmoothingLoss
from src.utils.plot_2 import plot_curves, plot_task2_comparison

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def setup_dirs():
    dirs = {"base": Path("outputs/task2")}
    # Updated: "checkpoints" replaced with "models" to match requested structure
    for name in ("logs", "plots", "results", "models", "wandb"):
        dirs[name] = Path("outputs/task2") / name
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)
    return dirs

def setup_logging(log_path):
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a"),
                  logging.StreamHandler(sys.stdout)])
    return logging.getLogger("Task2")

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
    decay, no_decay = [],[]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in['log_dt', 'B', 'C', 'D', 'bias', 'norm']):
            no_decay.append(param)
        else:
            decay.append(param)
    optim_groups =[
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=lr)

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

@torch.no_grad()
def generate_nwp(model, vocab, device, seeds, seq_len, n_words=15, mode="greedy", temperature=0.8, top_k=10):
    model.eval()
    results =[]
    for seed in seeds:
        tokens = seed.lower().split()
        ids = vocab.encode(tokens)
        ids = ([vocab.pad_idx] * (seq_len - len(ids)) + ids if len(ids) < seq_len else ids[-seq_len:])
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
def generate_mlm_samples(model, vocab, device, val_stream, seq_len, mask_prob, n=5):
    model.eval(); out =[]
    for i in range(n):
        ids = val_stream[i*seq_len : (i+1)*seq_len]
        if len(ids) < seq_len: break
        masked, mask_pos = list(ids),[]
        for j, tok in enumerate(ids):
            if random.random() < mask_prob:
                masked[j] = vocab.mask_idx; mask_pos.append(j)
        x = torch.tensor([masked], dtype=torch.long, device=device)
        preds = model(x).argmax(-1).squeeze(0).tolist()
        pred_sent = list(vocab.decode(ids))
        for pos in mask_pos: pred_sent[pos] = vocab.decode([preds[pos]])[0]
        out.append({"original": " ".join(vocab.decode(ids)),
                    "masked": " ".join(vocab.decode(masked)),
                    "predicted": " ".join(pred_sent)})
    return out

@torch.no_grad()
def score_next_words(model, vocab, device, context, seq_len):
    tokens = context.lower().split()
    ids = vocab.encode(tokens)
    ids = ([vocab.pad_idx]*(seq_len-len(ids))+ids if len(ids)<seq_len else ids[-seq_len:])
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(x)[:, -1, :] 
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    top5_vals, top5_ids = probs.topk(5)
    return[(vocab.decode([i.item()])[0], f"{v.item()*100:.2f}%") for v, i in zip(top5_vals, top5_ids)]

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

def run_ssm(vocab, train_sents, val_sents, test_sents, device, dirs, log, cfg, mode, run):
    log.info("=" * 60)
    log.info("TASK 2a — S4D (Diagonal State Space), Autoregressive NWP (v11)")
    log.info("=" * 60)
    
    train_ds = NWPStreamDataset(train_sents, vocab, cfg["seq_len"])
    val_ds = NWPStreamDataset(val_sents, vocab, cfg["seq_len"])
    test_ds = NWPStreamDataset(test_sents, vocab, cfg["seq_len"])
    
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    
    model = SSMLanguageModel(
        len(vocab), cfg["d_model"], cfg["d_state"],
        cfg["n_layers"], cfg["dropout"], vocab.pad_idx).to(device)
        
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"SSM params: {n_params:,}")
    
    optimizer = create_optimizer(model, cfg["lr"], cfg["weight_decay"])
    scheduler = get_warmup_cosine_scheduler(optimizer, cfg["warmup"], cfg["epochs"])
    ls_crit = LabelSmoothingLoss(len(vocab), cfg["label_smooth"], vocab.pad_idx)
    ce_crit = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    
    # Path updated to use "models"
    best_path = dirs["models"] / "ssm_nwp_best.pt"
    metrics_path = dirs["results"] / "task2_ssm_metrics.json"
    
    if mode in ["train", "both"]:
        best_val = float("inf")
        no_improve = 0
        tr_losses, vl_losses, tr_ppx, vl_ppx = [],[], [], []
        
        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()
            train_epoch(model, train_dl, optimizer, ls_crit, device, is_mlm=False)
            tl = eval_epoch(model, train_dl, ce_crit, device, is_mlm=False)
            vl = eval_epoch(model, val_dl, ce_crit, device, is_mlm=False)
            scheduler.step()
            
            tp, vp = perplexity(tl), perplexity(vl)
            tr_losses.append(tl); vl_losses.append(vl)
            tr_ppx.append(tp); vl_ppx.append(vp)
            lr_now = optimizer.param_groups[0]["lr"]
            
            log.info(f"[SSM] {epoch:>3}/{cfg['epochs']} | Train loss={tl:.4f} ppl={tp:.2f} | Val loss={vl:.4f} ppl={vp:.2f} | lr={lr_now:.2e} | {time.time()-t0:.1f}s")
            if run: run.log({"ssm/train_loss": tl, "ssm/val_loss": vl, "ssm/train_ppl": tp, "ssm/val_ppl": vp, "ssm/lr": lr_now, "ssm/epoch": epoch})
            
            if vl < best_val:
                best_val = vl; no_improve = 0
                torch.save({"model_state": model.state_dict(),
                            "config": dict(vocab_size=len(vocab), d_model=cfg["d_model"], d_state=cfg["d_state"], n_layers=cfg["n_layers"], dropout=cfg["dropout"], pad_idx=vocab.pad_idx)}, best_path)
                log.info(f" -> best model saved (val_loss={vl:.4f} ppl={vp:.2f})")
            else:
                no_improve += 1
                if no_improve >= cfg["patience"]:
                    log.info(f"[Early stop] {cfg['patience']} epochs no improvement.")
                    break
                    
        metrics = {"train_loss": tr_losses, "val_loss": vl_losses, "train_ppl": tr_ppx, "val_ppl": vl_ppx}
        with open(metrics_path, "w") as f: json.dump(metrics, f)
        
    if mode in["evaluate", "both"]:
        if not best_path.exists():
            log.info("[ERROR] SSM Checkpoint not found in models/. Cannot evaluate.")
            return
            
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
        test_loss = eval_epoch(model, test_dl, ce_crit, device, is_mlm=False)
        test_ppl = perplexity(test_loss)
        log.info(f"[SSM] TEST -> loss={test_loss:.4f} ppl={test_ppl:.2f}")
        
        if run: run.log({"ssm/test_loss": test_loss, "ssm/test_ppl": test_ppl})
        
        if metrics_path.exists():
            with open(metrics_path, "r") as f: metrics = json.load(f)
        else:
            metrics = {"train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl":[]}
            
        metrics["test_loss"] = test_loss
        metrics["test_ppl"] = test_ppl
        with open(metrics_path, "w") as f: json.dump(metrics, f)
        
        if metrics["train_loss"]:
            plot_curves(metrics["train_loss"], metrics["val_loss"], "Loss", "S4D — NWP Loss (v11)", dirs["plots"] / "ssm_loss.png")
            plot_curves(metrics["train_ppl"], metrics["val_ppl"], "Perplexity", "S4D — NWP Perplexity (v11)", dirs["plots"] / "ssm_perplexity.png")
        
        seeds =["the jury said", "the city of", "fulton county grand", "it recommended that", "the election was", "the president of the", "police department the jury"]
        gens_greedy = generate_nwp(model, vocab, device, seeds, cfg["seq_len"], n_words=15, mode="greedy")
        random.seed(cfg["seed"])
        gens_topk = generate_nwp(model, vocab, device, seeds, cfg["seq_len"], n_words=15, mode="topk", temperature=0.8, top_k=10)
        
        demo_contexts =["the fulton county grand jury said", "the city of atlanta for the", "it recommended that fulton legislators"]
        lines =["=" * 70, "S4D — Next-Word Prediction Results (v11)", "=" * 70,
                 f"Vocab size : {len(vocab)}", f"Parameters : {n_params:,}",
                 f"Architecture : S4D_Gated, d_model={cfg['d_model']}, d_state={cfg['d_state']}",
                 f"Best val loss : {min(metrics['val_loss']) if metrics['val_loss'] else 'N/A'}",
                 f"Test loss : {test_loss:.4f} (ppl={test_ppl:.2f})", "",
                 "─" * 70, "EXAMPLE 1: Greedy Continuations (+15 words)", "─" * 70]
        for seed, gen in zip(seeds, gens_greedy):
            lines += [f"Seed : {seed}", f"Output: {gen}", ""]
        lines +=["─" * 70, "EXAMPLE 2: Top-K Sampled Continuations (k=10, temp=0.8)", "─" * 70]
        for seed, gen in zip(seeds, gens_topk):
            lines +=[f"Seed : {seed}", f"Output: {gen}", ""]
        lines +=["─" * 70, "EXAMPLE 3: Top-5 Most Likely Next Words (word scoring)", "─" * 70]
        for ctx in demo_contexts:
            top5 = score_next_words(model, vocab, device, ctx, cfg["seq_len"])
            lines +=[f"Context : '{ctx}'", f"Top-5 : {', '.join(f'{w}({p})' for w, p in top5)}", ""]
            
        (dirs["results"] / "task2_ssm.txt").write_text("\n".join(lines))

def run_bilstm(vocab, train_sents, val_sents, test_sents, device, dirs, log, cfg, mode, run):
    log.info("=" * 60)
    log.info("TASK 2b — Bi-LSTM MLM (v11, Custom Recurrence)")
    log.info("=" * 60)
    
    train_ds = MLMStreamDataset(train_sents, vocab, cfg["seq_len"], cfg["stride"], cfg["mask_prob"])
    val_ds = MLMStreamDataset(val_sents, vocab, cfg["seq_len"], cfg["stride"], cfg["mask_prob"])
    test_ds = MLMStreamDataset(test_sents, vocab, cfg["seq_len"], cfg["stride"], cfg["mask_prob"])
    val_stream = vocab.encode(sentences_to_stream(val_sents, vocab.EOS))
    
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    
    model = BiLSTMMLM(
        len(vocab), cfg["embed_dim"], cfg["hidden_dim"],
        cfg["n_layers"], cfg["dropout"], vocab.pad_idx).to(device)
        
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Bi-LSTM params: {n_params:,}")
    
    optimizer = create_optimizer(model, cfg["lr"], cfg["weight_decay"])
    scheduler = get_warmup_cosine_scheduler(optimizer, cfg["warmup"], cfg["epochs"])
    ls_crit = LabelSmoothingLoss(len(vocab), cfg["label_smooth"], -100)
    ce_crit = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Path updated to use "models"
    best_path = dirs["models"] / "bilstm_mlm_best.pt"
    metrics_path = dirs["results"] / "task2_bilstm_metrics.json"
    
    if mode in ["train", "both"]:
        best_val = float("inf")
        no_improve = 0
        tr_losses, vl_losses, tr_ppx, vl_ppx = [], [], [], []
        
        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()
            train_epoch(model, train_dl, optimizer, ls_crit, device, is_mlm=True)
            tl = eval_epoch(model, train_dl, ce_crit, device, is_mlm=True)
            vl = eval_epoch(model, val_dl, ce_crit, device, is_mlm=True)
            scheduler.step()
            
            tp, vp = perplexity(tl), perplexity(vl)
            tr_losses.append(tl); vl_losses.append(vl)
            tr_ppx.append(tp); vl_ppx.append(vp)
            lr_now = optimizer.param_groups[0]["lr"]
            
            log.info(f"[BiLSTM] {epoch:>3}/{cfg['epochs']} | Train loss={tl:.4f} ppl={tp:.2f} | Val loss={vl:.4f} ppl={vp:.2f} | lr={lr_now:.2e} | {time.time()-t0:.1f}s")
            if run: run.log({"bilstm/train_loss": tl, "bilstm/val_loss": vl, "bilstm/train_ppl": tp, "bilstm/val_ppl": vp, "bilstm/lr": lr_now, "bilstm/epoch": epoch})
            
            if vl < best_val:
                best_val = vl; no_improve = 0
                torch.save({"model_state": model.state_dict(),
                            "config": dict(vocab_size=len(vocab), embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"], dropout=cfg["dropout"], pad_idx=vocab.pad_idx)}, best_path)
                log.info(f" -> best model saved (val_loss={vl:.4f} ppl={vp:.2f})")
            else:
                no_improve += 1
                if no_improve >= cfg["patience"]:
                    log.info(f" [Early stop] {cfg['patience']} epochs no improvement.")
                    break
                    
        metrics = {"train_loss": tr_losses, "val_loss": vl_losses, "train_ppl": tr_ppx, "val_ppl": vl_ppx}
        with open(metrics_path, "w") as f: json.dump(metrics, f)
        
    if mode in ["evaluate", "both"]:
        if not best_path.exists():
            log.info("[ERROR] BiLSTM Checkpoint not found in models/. Cannot evaluate.")
            return
            
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
        test_loss = eval_epoch(model, test_dl, ce_crit, device, is_mlm=True)
        test_ppl = perplexity(test_loss)
        log.info(f"[BiLSTM] TEST -> loss={test_loss:.4f} ppl={test_ppl:.2f}")
        
        if run: run.log({"bilstm/test_loss": test_loss, "bilstm/test_ppl": test_ppl})
        
        if metrics_path.exists():
            with open(metrics_path, "r") as f: metrics = json.load(f)
        else:
            metrics = {"train_loss":[], "val_loss": [], "train_ppl": [], "val_ppl": []}
            
        metrics["test_loss"] = test_loss
        metrics["test_ppl"] = test_ppl
        with open(metrics_path, "w") as f: json.dump(metrics, f)
        
        if metrics["train_loss"]:
            plot_curves(metrics["train_loss"], metrics["val_loss"], "Loss", "Bi-LSTM — MLM Loss (v11)", dirs["plots"] / "bilstm_loss.png")
            plot_curves(metrics["train_ppl"], metrics["val_ppl"], "Perplexity", "Bi-LSTM — MLM Perplexity (v11)", dirs["plots"] / "bilstm_perplexity.png")
        
        samples = generate_mlm_samples(model, vocab, device, val_stream, cfg["seq_len"], cfg["mask_prob"], n=5)
        
        demo_sentences =[
            "the fulton county grand jury said <mask> an investigation",
            "the city of <mask> for the manner in which the election",
            "it recommended that fulton <mask> act to have these laws",
            "the jury said it did find that many of <mask> registration",
            "merger <mask> however the jury said it believes",
        ]
        
        lines =["=" * 70, "Bi-LSTM — Masked Language Modeling Results (v11)", "=" * 70,
                 f"Vocab size : {len(vocab)}", f"Parameters : {n_params:,}",
                 f"Architecture : Custom BiLSTM, hidden={cfg['hidden_dim']}, layers={cfg['n_layers']}",
                 f"Best val loss : {min(metrics['val_loss']) if metrics['val_loss'] else 'N/A'}",
                 f"Test loss : {test_loss:.4f} (ppl={test_ppl:.2f})", "",
                 "─" * 70, "EXAMPLE 1: Random Masking (15% of tokens) — Fill-in-the-Blank", "─" * 70]
        for i, s in enumerate(samples, 1):
            lines += [f"Sample {i}", f" Original : {s['original']}", f" Masked : {s['masked']}", f" Predicted: {s['predicted']}", ""]
        lines +=["─" * 70, "EXAMPLE 2: Targeted Single-Word Fill (shows bidirectional context)", "─" * 70]
        for sent in demo_sentences:
            results = fill_single_mask(model, vocab, device, sent, cfg["seq_len"])
            lines += [f"Sentence : {sent}"]
            for pos, top5 in results: lines +=[f" Mask at pos {pos}: {', '.join(f'{w}({p})' for w, p in top5)}"]
            lines += [""]
            
        (dirs["results"] / "task2_bilstm.txt").write_text("\n".join(lines))

def check_and_run_comparison(dirs, log):
    ssm_p = dirs["results"] / "task2_ssm_metrics.json"
    bilstm_p = dirs["results"] / "task2_bilstm_metrics.json"
    
    if ssm_p.exists() and bilstm_p.exists():
        with open(ssm_p, "r") as f: ssm_m = json.load(f)
        with open(bilstm_p, "r") as f: bilstm_m = json.load(f)
        
        lines =[
            "=" * 70, "TASK 2 — Model Comparison (v11)", "=" * 70,
            f"{'Metric':<30} {'S4D (NWP)':>15} {'Bi-LSTM (MLM)':>15}", "-" * 62,
            f"{'Best Val Loss':<30} {min(ssm_m['val_loss']) if ssm_m['val_loss'] else 'N/A':>15} {min(bilstm_m['val_loss']) if bilstm_m['val_loss'] else 'N/A':>15}",
            f"{'Best Val Perplexity':<30} {min(ssm_m['val_ppl']) if ssm_m['val_ppl'] else 'N/A':>15} {min(bilstm_m['val_ppl']) if bilstm_m['val_ppl'] else 'N/A':>15}",
            f"{'Test Loss':<30} {ssm_m.get('test_loss', 0):>15.4f} {bilstm_m.get('test_loss', 0):>15.4f}",
            f"{'Test Perplexity':<30} {ssm_m.get('test_ppl', 0):>15.2f} {bilstm_m.get('test_ppl', 0):>15.2f}",
            "",
            "Notes:",
            " S4D : Causal autoregressive NWP using FFT Convolution over frozen HiPPO/LegS complex state.",
            " Bi-LSTM: Custom bidirectional MLM, hidden=256, seq_len=128.",
        ]
        (dirs["results"] / "task2_comparison.txt").write_text("\n".join(lines))
        plot_task2_comparison(ssm_m, bilstm_m, dirs["plots"] / "task2_comparison.png")
        log.info("[COMPARISON] Summary and plot saved.")

def run_task(kind, cfg, mode):
    set_seed(cfg["seed"])
    device = get_device()
    dirs = setup_dirs()
    log = setup_logging(dirs["logs"] / f"task2_{kind}.log")
    log.info(f"Data: {cfg['data_path']} | Output: outputs/task2 | Device: {device}")
    
    run = None
    try:
        wandb.login(key=cfg["wandb_api_key"], relogin=True)
        run = wandb.init(
            project=cfg["wandb_project"], 
            name=f"{'SSM_NWP' if kind == 'ssm' else 'BiLSTM_MLM'}_v11", 
            dir="outputs/task2",
            config=cfg
        )
    except Exception as e:
        log.info(f"[WARN] wandb unavailable: {e}")

    sentences = load_and_tokenize(cfg["data_path"])
    train_sents, val_sents, test_sents = split_sentences(sentences, cfg["train_split"], cfg["val_split"])
    log.info(f"Split -> train={len(train_sents)} val={len(val_sents)} test={len(test_sents)}")
    
    train_stream = sentences_to_stream(train_sents, "<EOS>")
    vocab = Vocabulary(min_freq=cfg["min_freq"])
    vocab.build(train_stream)
    
    # Path updated to use "models"
    (dirs["models"] / "vocab.json").write_text(json.dumps({
        "word2idx": vocab.word2idx,
        "idx2word": {str(k): v for k, v in vocab.idx2word.items()}}, indent=2))
        
    if kind == "ssm":
        run_ssm(vocab, train_sents, val_sents, test_sents, device, dirs, log, cfg, mode, run)
    elif kind == "bilstm":
        run_bilstm(vocab, train_sents, val_sents, test_sents, device, dirs, log, cfg, mode, run)
        
    check_and_run_comparison(dirs, log)
    
    if run: run.finish()