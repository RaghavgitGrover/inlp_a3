import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.utils.metrics import levenshtein, char_acc

def plot_training(history, tag):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12,4))
    a1.plot(history["train_loss"], label="Train")
    a1.plot(history["val_loss"],   label="Val")
    a1.set_title(f"{tag.upper()} Loss"); 
    a1.legend(); 
    a1.grid(alpha=.3)
    a2.plot(history["val_char_acc"], color="green")
    a2.set_title(f"{tag.upper()} Val Char Accuracy"); 
    a2.grid(alpha=.3)
    p = Path(f"outputs/task1/plots/task1_{tag}_training.png")
    fig.tight_layout(); 
    fig.savefig(p, dpi=150); 
    plt.close(fig); 
    return p

def plot_lev(preds, refs, tag):
    d = [levenshtein(p,r) for p,r in zip(preds,refs)]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(d, bins=30, color="#4C72B0", edgecolor="white")
    ax.axvline(np.mean(d), color="red", ls="--", label=f"Mean={np.mean(d):.1f}")
    ax.set_title(f"{tag.upper()} Edit Distance (Test)"); 
    ax.legend(); 
    ax.grid(alpha=.3)
    p = Path(f"outputs/task1/plots/task1_{tag}_lev.png")
    fig.tight_layout(); 
    fig.savefig(p, dpi=150); 
    plt.close(fig); 
    return p

def plot_acc(preds, refs, tag):
    a =[char_acc(p,r) for p,r in zip(preds,refs)]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(a, bins=20, range=(0,1), color="#DD8452", edgecolor="white")
    ax.axvline(np.mean(a), color="red", ls="--", label=f"Mean={np.mean(a):.3f}")
    ax.set_title(f"{tag.upper()} Char Accuracy (Test)"); 
    ax.legend(); 
    ax.grid(alpha=.3)
    p = Path(f"outputs/task1/plots/task1_{tag}_char_acc.png")
    fig.tight_layout(); 
    fig.savefig(p, dpi=150); 
    plt.close(fig); 
    return p

def plot_cmp(rm, lm):
    keys = list(rm.keys()); 
    x = np.arange(len(keys)); 
    w = .35
    fig, ax = plt.subplots(figsize=(9,5))
    b1 = ax.bar(x-w/2, [rm[k] for k in keys], w, label="RNN",  color="#4C72B0")
    b2 = ax.bar(x+w/2, [lm[k] for k in keys], w, label="LSTM", color="#DD8452")
    ax.set_xticks(x); 
    ax.set_xticklabels([k.replace("_","\n") for k in keys], fontsize=9)
    ax.set_title("Task 1 — RNN vs LSTM"); 
    ax.legend(); 
    ax.grid(axis="y", alpha=.3)
    for bar in list(b1)+list(b2):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha="center", fontsize=8)
    p = Path("outputs/task1/plots/task1_comparison.png")
    fig.tight_layout(); 
    fig.savefig(p, dpi=150); 
    plt.close(fig); 
    return p