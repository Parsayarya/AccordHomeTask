import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

df = pd.read_csv("outputs/gemini_classification_candidates.csv")

CONF_COL = "predicted confidence"
LABEL_COL = "label"

n = len(df)
n_correct = int(df[LABEL_COL].sum())
n_incorrect = n - n_correct
accuracy = n_correct / n
error_rate = 1 - accuracy

from math import sqrt
z = 1.96
p = accuracy
den = 1 + z**2/n
centre = p + z**2/(2*n)
margin = z * sqrt((p*(1-p) + z**2/(4*n)) / n)
acc_ci_low = (centre - margin) / den
acc_ci_high = (centre + margin) / den


brier = np.mean((df[CONF_COL] - df[LABEL_COL])**2)

# AUC for "predicting correctness"
auc = roc_auc_score(df[LABEL_COL], df[CONF_COL])
ap  = average_precision_score(df[LABEL_COL], df[CONF_COL])


# Expected Calibration Error (ECE) with 10 bins
def expected_calibration_error(scores: pd.Series, labels: pd.Series, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(scores, bins, right=True)
    ece = 0.0
    for b in range(1, n_bins + 1):
        mask = (idx == b)
        if not np.any(mask):
            continue
        conf_bin = scores[mask].mean()
        acc_bin = labels[mask].mean()
        weight = mask.mean()
        ece += weight * abs(acc_bin - conf_bin)
    return ece

ece_10 = expected_calibration_error(df[CONF_COL], df[LABEL_COL], n_bins=10)

# Confidence 
overall_conf = df[CONF_COL].mean()
conf_when_correct = df.loc[df[LABEL_COL] == 1, CONF_COL].mean()
conf_when_wrong   = df.loc[df[LABEL_COL] == 0, CONF_COL].mean()

summary = {
    "n": n,
    "n_correct": n_correct,
    "n_incorrect": n_incorrect,
    "accuracy": round(accuracy, 4),
    "accuracy_95ci": (round(acc_ci_low, 4), round(acc_ci_high, 4)),
    "error_rate": round(error_rate, 4),
    "brier_score": round(brier, 4),
    "auc_correctness": None if np.isnan(auc) else round(auc, 4),
    "average_precision_correctness": None if np.isnan(ap) else round(ap, 4),
    "ece_10bins": round(ece_10, 4),
    "mean_conf_overall": round(overall_conf, 4),
    "mean_conf_when_correct": round(conf_when_correct, 4),
    "mean_conf_when_wrong": round(conf_when_wrong, 4),
}
print(summary)

# Distribution of predicted confidence per label

plt.figure(figsize=(8, 5))
for lbl in (0, 1):
    subset = df.loc[df[LABEL_COL] == lbl, CONF_COL].dropna()
    plt.hist(
        subset,
        bins=20,
        alpha=0.6,
        density=True,
        label=f"label={lbl}"
    )
plt.xlabel("Predicted confidence")
plt.ylabel("Density")
plt.title("Predicted confidence by correctness label")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
data = [df.loc[df[LABEL_COL] == 0, CONF_COL].dropna(),
        df.loc[df[LABEL_COL] == 1, CONF_COL].dropna()]
plt.boxplot(data, labels=["label=0 (wrong)", "label=1 (correct)"], showmeans=True)
plt.ylabel("Predicted confidence")
plt.title("Confidence distribution by label")
plt.tight_layout()
plt.show()

bins = pd.cut(df[CONF_COL], bins=np.linspace(0, 1, 11), include_lowest=True)
calib_tbl = (
    df.groupby(bins)
      .agg(bin_count=(LABEL_COL, "size"),
           avg_conf=(CONF_COL, "mean"),
           emp_acc=(LABEL_COL, "mean"))
      .reset_index()
)
print(calib_tbl)
