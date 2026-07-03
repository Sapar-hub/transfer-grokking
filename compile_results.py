"""
Compile multi-seed results into mean +/- std tables and paper figures.

Reads all seed outputs from:
    artifacts/ce_projection/seeds[,_mult]/*_seed*.csv
    artifacts/l31_patch[,_mult]/alpha_sweep_l31_seed*.csv
    artifacts/l31_patch[,_mult]/perplexity_sweep_seed*.csv

Generates:
    artifacts/paper_tables.md         -- all summary tables
    artifacts/paper_figures/          -- publication-ready PNGs
"""
import numpy as np
import os, csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARTIFACTS = "artifacts"
OUT_DIR = f"{ARTIFACTS}/paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
PPL_ALPHAS = [0.0, 0.5, 1.0]


def read_csv_rows(path):
    with open(path) as f:
        reader = csv.reader(f)
        return list(reader)


def main():
    tasks = [("add", ""), ("multiply", "_mult")]
    task_labels = {"add": "Addition", "multiply": "Multiplication"}

    # --- Collect per-task per-seed CE L31 sweep data ---
    ce_l31_data = {}
    for task_name, suffix in tasks:
        seed_vals = []
        for seed in SEEDS:
            path = f"{ARTIFACTS}/l31_patch{suffix}/alpha_sweep_l31_seed{seed}.csv"
            rows = read_csv_rows(path)
            for row in rows:
                if row[0] == "CE":
                    vals = [float(v) for v in row[1:]]
                    seed_vals.append(vals)
                    break
        ce_l31_data[task_name] = np.array(seed_vals)

    # --- Generate tables ---
    lines = []
    lines.append("# Multi-Seed Results (mean +/- std, n=5)\n")
    lines.append(f"_Seeds: {SEEDS}_\n")

    # Table 1: L31 Alpha Sweep -- Accuracy
    lines.append("## 1. L31 Alpha Sweep -- Accuracy\n")
    lines.append("| Alpha | Addition (W_CE L31) | Multiplication (W_CE L31) |")
    lines.append("|-------|---------------------|---------------------------|")

    for i, alpha in enumerate(ALPHAS):
        add_mean = ce_l31_data["add"][:, i].mean()
        add_std = ce_l31_data["add"][:, i].std()
        mult_mean = ce_l31_data["multiply"][:, i].mean()
        mult_std = ce_l31_data["multiply"][:, i].std()
        lines.append(f"| {alpha:.1f} | {add_mean:.4f} +/- {add_std:.4f} | {mult_mean:.4f} +/- {mult_std:.4f} |")
    lines.append("")
    lines.append("Note: Baseline (alpha=0.0) is Phi-2 LM head accuracy without patch (no seed variance).\n")

    # Table 2: Perplexity
    lines.append("## 2. L31 Patch -- Perplexity (WikiText-2 last-token)\n")
    lines.append("| Alpha | Addition | | Multiplication | |")
    lines.append("|-------|----------|--|----------------|--|")
    lines.append("| | Loss | PPL | Loss | PPL |")

    for j, alpha in enumerate(PPL_ALPHAS):
        add_losses, add_ppls = [], []
        mult_losses, mult_ppls = [], []
        for seed in SEEDS:
            for task_name, suffix, losses, ppls in [
                ("add", "", add_losses, add_ppls),
                ("multiply", "_mult", mult_losses, mult_ppls),
            ]:
                path = f"{ARTIFACTS}/l31_patch{suffix}/perplexity_sweep_seed{seed}.csv"
                rows = read_csv_rows(path)
                losses.append(float(rows[1 + j][1]))
                ppls.append(float(rows[1 + j][2]))

        add_l = np.mean(add_losses), np.std(add_losses)
        add_p = np.mean(add_ppls), np.std(add_ppls)
        mult_l = np.mean(mult_losses), np.std(mult_losses)
        mult_p = np.mean(mult_ppls), np.std(mult_ppls)

        lines.append(
            f"| {alpha:.1f} | {add_l[0]:.4f}+/-{add_l[1]:.4f} | "
            f"{add_p[0]:.2f}+/-{add_p[1]:.2f} | "
            f"{mult_l[0]:.4f}+/-{mult_l[1]:.4f} | "
            f"{mult_p[0]:.2f}+/-{mult_p[1]:.2f} |"
        )
    lines.append("")

    # Table 3: Summary at alpha=0.5 (sweet spot)
    lines.append("## 3. Summary at alpha=0.5 (sweet spot)\n")
    lines.append("| Task | Acc@0.5 | PPL@0.5 | PPL@0.0 (baseline) | PPL@1.0 |")
    lines.append("|------|---------|---------|-------------------|---------|")
    for task_name, suffix in tasks:
        acc_i = ALPHAS.index(0.5)
        acc_mean = ce_l31_data[task_name][:, acc_i].mean()
        acc_std = ce_l31_data[task_name][:, acc_i].std()

        seed_ppls = {a: [] for a in PPL_ALPHAS}
        for seed in SEEDS:
            p = f"{ARTIFACTS}/l31_patch{suffix}/perplexity_sweep_seed{seed}.csv"
            rows = read_csv_rows(p)
            for r in rows[1:]:
                a = float(r[0])
                ppl = float(r[2])
                seed_ppls[a].append(ppl)

        ppl_05 = np.mean(seed_ppls[0.5]), np.std(seed_ppls[0.5])
        ppl_00 = np.mean(seed_ppls[0.0]), np.std(seed_ppls[0.0])
        ppl_10 = np.mean(seed_ppls[1.0]), np.std(seed_ppls[1.0])

        lines.append(
            f"| {task_labels[task_name]} | {acc_mean:.4f}+/-{acc_std:.4f} | "
            f"{ppl_05[0]:.2f}+/-{ppl_05[1]:.2f} | "
            f"{ppl_00[0]:.2f}+/-{ppl_00[1]:.2f} | "
            f"{ppl_10[0]:.2f}+/-{ppl_10[1]:.2f} |"
        )

    table_text = "\n".join(lines)
    with open(f"{ARTIFACTS}/paper_tables.md", "w") as f:
        f.write(table_text)
    print(table_text)

    # --- FIGURES ---

    # Figure: Alpha sweep twin-axis (accuracy + perplexity)
    for task_name, suffix in tasks:
        fig, ax1 = plt.subplots(figsize=(8, 5))

        acc_means = ce_l31_data[task_name].mean(axis=0)
        acc_stds = ce_l31_data[task_name].std(axis=0)
        color_acc = 'tab:blue'
        ax1.set_xlabel('Alpha (injection strength)')
        ax1.set_ylabel('Accuracy', color=color_acc)
        ax1.plot(ALPHAS, acc_means, 'o-', color=color_acc, linewidth=2, markersize=8, label='Accuracy')
        ax1.fill_between(ALPHAS, acc_means - acc_stds, acc_means + acc_stds, alpha=0.2, color=color_acc)
        ax1.tick_params(axis='y', labelcolor=color_acc)
        ax1.axhline(y=1/97, color='gray', linestyle='--', alpha=0.4)
        ax1.set_ylim(-0.05, 1.05)

        seed_ppls = {a: [] for a in PPL_ALPHAS}
        for seed in SEEDS:
            p = f"{ARTIFACTS}/l31_patch{suffix}/perplexity_sweep_seed{seed}.csv"
            rows = read_csv_rows(p)
            for r in rows[1:]:
                a = float(r[0])
                ppl = float(r[2])
                seed_ppls[a].append(ppl)

        ppl_alphas = sorted(seed_ppls.keys())
        ppl_means = np.array([np.mean(seed_ppls[a]) for a in ppl_alphas])
        ppl_stds = np.array([np.std(seed_ppls[a]) for a in ppl_alphas])

        ax2 = ax1.twinx()
        color_ppl = 'tab:red'
        ax2.set_ylabel('Perplexity (log scale)', color=color_ppl)
        ax2.semilogy(ppl_alphas, ppl_means, 's--', color=color_ppl, linewidth=2, markersize=8, label='Perplexity')
        ax2.fill_between(ppl_alphas, ppl_means - ppl_stds, ppl_means + ppl_stds, alpha=0.2, color=color_ppl)
        ax2.tick_params(axis='y', labelcolor=color_ppl)

        fig.suptitle(f'{task_labels[task_name]} mod 97 -- L31 Patch (n=5 seeds)')
        fig.tight_layout()
        path = f"{OUT_DIR}/alpha_sweep_{task_name}.png"
        fig.savefig(path, dpi=150)
        plt.close()
        print(f"  Figure saved: {path}")

    # Cross-task comparison: addition vs multiplication accuracy only
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (task_name, suffix, title, color) in zip(
        [ax1, ax2],
        [("add", "", "Addition mod 97", "tab:blue"),
         ("multiply", "_mult", "Multiplication mod 97", "tab:orange")]
    ):
        acc_means = ce_l31_data[task_name].mean(axis=0)
        acc_stds = ce_l31_data[task_name].std(axis=0)

        ax.set_xlabel('Alpha')
        ax.set_ylabel('Accuracy')
        ax.plot(ALPHAS, acc_means, 'o-', color=color, linewidth=2, markersize=8)
        ax.fill_between(ALPHAS, acc_means - acc_stds, acc_means + acc_stds, alpha=0.2, color=color)
        ax.axhline(y=1/97, color='gray', linestyle='--', alpha=0.4, label='random')
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

        for i, (m, s) in enumerate(zip(acc_means, acc_stds)):
            ax.annotate(f'{m:.3f}', (ALPHAS[i], m + 0.05), ha='center', fontsize=8)

    fig.suptitle('Cross-task Comparison: W_CE L31 Patch (mean +/- std, n=5)')
    fig.tight_layout()
    path = f"{OUT_DIR}/cross_task_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Figure saved: {path}")

    print(f"\nAll results compiled to {OUT_DIR}/ and {ARTIFACTS}/paper_tables.md")


if __name__ == "__main__":
    main()
