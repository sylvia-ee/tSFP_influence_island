
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

def plot_policy_heatmaps(optimal_Q):

    rounds = sorted(optimal_Q["round"].unique())

    for r in rounds:

        df = optimal_Q[optimal_Q["round"] == r].copy()

        # pivot tbl for heatmap
        actions = sorted(optimal_Q["action"].unique())
        action_map = {a: i for i, a in enumerate(actions)}
        df["action_id"] = df["action"].map(action_map)

        df = df.sort_values(["trial", "vs_left", "score"])

        scores = sorted(df["score"].unique())
        trials = sorted(df["trial"].unique())
        vs_vals = sorted(df["vs_left"].unique())

        col_order = [(t, vs) for t in trials for vs in vs_vals]

        pivot = df.pivot(
            index="score",
            columns=["trial", "vs_left"],
            values="action_id"
        )

        pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(col_order))

        Z = pivot.values

        # resize to reduce visual noise
        fig, ax = plt.subplots(
            figsize=(max(12, Z.shape[1]*0.35),
                     max(6, Z.shape[0]*0.12))
        )

        # colormap
        cmap = ListedColormap(plt.cm.tab10.colors[:len(actions)])
        norm = BoundaryNorm(np.arange(len(actions)+1)-0.5, cmap.N)

        mesh = ax.pcolormesh(
            np.arange(Z.shape[1] + 1),
            np.arange(Z.shape[0] + 1),
            Z,
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="black",
            linewidth=0.4
        )

        # axes
        ax.set_title(f"Optimal Policy (Round {r})", pad=40)
        ax.set_ylabel("Score")
        ax.set_xlabel("vs_left (nested within trial)")

        x_centers = np.arange(Z.shape[1]) + 0.5
        vs_labels = [vs for (_, vs) in col_order]
        ax.set_xticks(x_centers)
        ax.set_xticklabels(vs_labels)

        y_centers = np.arange(Z.shape[0]) + 0.5
        ax.set_yticks(y_centers[::10])
        ax.set_yticklabels(scores[::10])

        # group by trial
        n_vs = len(vs_vals)

        for i, t in enumerate(trials):
            start = i * n_vs
            ax.axvline(start, color="black", linewidth=1.5)

            center = start + n_vs / 2
            ax.text(center, Z.shape[0] + 1.5, f"T{t}",
                    ha="center", va="bottom", fontsize=10)

        ax.axvline(Z.shape[1], color="black", linewidth=1.5)

        # add in ranges for win and convince

        for i, t in enumerate(trials):

            rule_row = df[df["trial"] == t].iloc[0]

            win_low = rule_row["win_low"]
            win_high = rule_row["win_high"]
            conv_low = rule_row["conv_low"]
            conv_high = rule_row["conv_high"]

            start = i * n_vs
            end = start + n_vs

            # align with grid by +0.5
            if pd.notna(win_low):
                ax.plot([start, end], [win_low, win_low],
                        linestyle="--", color="black", linewidth=1.2)
                ax.plot([start, end], [win_high, win_high],
                        linestyle="--", color="black", linewidth=1.2)

            if pd.notna(conv_low):
                ax.plot([start, end], [conv_low, conv_low],
                        linestyle=":", color="black", linewidth=1.2)
                ax.plot([start, end], [conv_high, conv_high],
                        linestyle=":", color="black", linewidth=1.2)
                
        # color bar for legend decisions
        cbar = plt.colorbar(mesh, ax=ax, pad=0.02)

        cbar.set_ticks(range(len(actions)))
        cbar.set_ticklabels(actions)
        cbar.set_label("Action")


        # legend for lines
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='--',
                   label='Win Range'),
            Line2D([0], [0], color='black', linestyle=':',
                   label='Convince Range')
        ]

        # legend for decisions
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.18, 1),  
            loc="upper left",
            borderaxespad=0,
            frameon=False
        )

        # add whitespace on R
        plt.subplots_adjust(right=0.78, top=0.85)
