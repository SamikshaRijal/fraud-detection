"""Member 3 estimators and training-only Random Forest importance plot."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


class BalancedGradientBoostingClassifier(GradientBoostingClassifier):
    """Compute inverse-frequency weights inside each CV fit, never on validation data."""

    def fit(self, X, y, sample_weight=None, monitor=None):
        weights = compute_sample_weight("balanced", y)
        if sample_weight is not None:
            weights = weights * sample_weight
        return super().fit(X, y, sample_weight=weights, monitor=monitor)


def save_feature_importance(pipeline, name, top_n=15):
    """Use transformer output names so importance labels match model inputs."""
    names = pipeline.named_steps["scale"].get_feature_names_out()
    frame = pd.DataFrame({
        "Feature": [name.split("__", 1)[-1] for name in names],
        "Importance": pipeline.named_steps["model"].feature_importances_,
    }).sort_values("Importance", ascending=False)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    frame.to_csv("results/metrics/feature_importance.csv", index=False)
    shown = frame.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(shown["Feature"], shown["Importance"], color="#2563a6")
    ax.set_xlabel("Mean decrease in impurity (normalized importance)")
    ax.set_title(f"Top {top_n} features — {name}", pad=15)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.015, "Forest selected by training CV PR-AUC. V1–V28 are anonymized PCA components.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig("results/figures/feature_importance.png", dpi=180)
    plt.close(fig)
    return frame
