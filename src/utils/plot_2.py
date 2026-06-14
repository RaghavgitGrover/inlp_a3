import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_curves(train_vals, val_vals, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(train_vals) + 1)
    ax.plot(epochs, train_vals, label=f"Train {ylabel}", linewidth=2)
    ax.plot(epochs, val_vals, label=f"Val {ylabel}", linewidth=2)
    ax.set_title(title, fontsize=13); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_task2_comparison(sm, bm, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models =["S4D (NWP)", "Bi-LSTM (MLM)"]
    colors =["#4C72B0", "#DD8452"]
    
    val_ppl_s = min(sm["val_ppl"]) if sm.get("val_ppl") else sm.get("test_ppl", 0)
    val_ppl_b = min(bm["val_ppl"]) if bm.get("val_ppl") else bm.get("test_ppl", 0)
    
    metrics =[
        ([val_ppl_s, val_ppl_b], "Best Validation Perplexity"),
        ([sm["test_ppl"], bm["test_ppl"]], "Test Perplexity")
    ]
    
    for ax, vals, title in zip(axes, *zip(*metrics)):
        bars = ax.bar(models, vals, color=colors, width=0.5)
        ax.set_title(title, fontsize=12); ax.set_ylabel("Perplexity")
        for bar, v in zip(bars, vals): 
            ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")
            
    plt.suptitle("Task 2 — S4D vs Bi-LSTM (v11)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)