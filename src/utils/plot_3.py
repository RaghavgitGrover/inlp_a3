import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

def make_plots(all_results: dict, noise_levels: list, out_dir):
    colors = {"LSTM only": "#4C72B0", "LSTM + SSM": "#DD8452", "LSTM + BiLSTM": "#55A868"}
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    metric_list =[
        ("word_accuracy",        "Word Accuracy"),
        ("char_accuracy",        "Char Accuracy"),
        ("levenshtein_distance", "Levenshtein Distance"),
        ("rouge1",               "ROUGE-1"),
        ("rouge2",               "ROUGE-2"),
        ("rougeL",               "ROUGE-L"),
        ("bleu1",                "BLEU-1"),
        ("bleu4",                "BLEU-4"),
    ]
    w, xs = 0.25, np.arange(len(noise_levels))

    for ax, (key, label) in zip(axes.flatten(), metric_list):
        l_vals =[all_results[x]["lstm"].get(key, 0) for x in noise_levels]
        s_vals = [all_results[x]["ssm"].get(key, 0) for x in noise_levels]
        b_vals = [all_results[x]["bilstm"].get(key, 0) for x in noise_levels]
        b1 = ax.bar(xs - w, l_vals, w, label="LSTM only", color=colors["LSTM only"])
        b2 = ax.bar(xs, s_vals, w, label="LSTM + SSM", color=colors["LSTM + SSM"])
        b3 = ax.bar(xs + w, b_vals, w, label="LSTM + BiLSTM", color=colors["LSTM + BiLSTM"])
        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
        ax.set_xticks(xs); ax.set_xticklabels([f"Noise {n}" for n in noise_levels])
        ax.set_title(label, fontsize=11); ax.set_ylabel(label); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Task 3 — LM-Assisted Decryption Error Correction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = out_dir / "task3_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    wandb.log({"task3_comparison_plot": wandb.Image(str(plot_path))})