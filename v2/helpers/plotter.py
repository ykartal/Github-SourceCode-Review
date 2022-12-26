import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn")

def visualize_results(results: pd.DataFrame, x: str, y: str, kind: str, title: str, x_label: str, y_label: str, legend_keys: list[str], save=False, path=None):
    ax = results.plot(x=x, y=y, kind=kind, figsize=(8, 5),
                      fontsize=12, linewidth=3)
    plt.tight_layout()
    ax.legend(legend_keys, fontsize=12, frameon=True)
    ax.set_title(title, fontsize=15)
    ax.invert_xaxis()
    plt.ylabel(y_label, fontsize=13)
    plt.xlabel(x_label, fontsize=13)
    if save:
        plt.savefig(path,
                    facecolor="w", edgecolor="w", bbox_inches="tight")
