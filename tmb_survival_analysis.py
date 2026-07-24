"""
TMB vs Survival Analysis
===========================

pip install pandas lifelines matplotlib
"""

import glob
import json
import os

import pandas as pd
import matplotlib.pyplot as plt

WORK_DIR = "maf_pipeline"
RESULTS_DIR = os.path.join(WORK_DIR, "results")
DOWNSTREAM_DIR = os.path.join(WORK_DIR, "downstream")
TMB_PATH = os.path.join(DOWNSTREAM_DIR, "tmb_per_sample.csv")


def find_one(pattern):
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file found matching: {pattern}")
    return matches[0]


def load_clinical_with_survival():
    path = find_one("clinical.cohort*.json")
    with open(path) as f:
        records = json.load(f)

    rows = []
    for rec in records:
        demographic = rec.get("demographic") or {}
        diagnoses = rec.get("diagnoses") or [{}]
        primary = diagnoses[0] if diagnoses else {}

        rows.append({
            "submitter_id": rec.get("submitter_id"),
            "days_to_death": demographic.get("days_to_death"),
            "days_to_last_follow_up": demographic.get("days_to_last_follow_up")
                or primary.get("days_to_last_follow_up"),
            "age_at_index": demographic.get("age_at_index"),
        })

    clinical = pd.DataFrame(rows)
    clinical["time"] = clinical["days_to_death"].fillna(clinical["days_to_last_follow_up"])
    clinical["event"] = clinical["days_to_death"].notna().astype(int)
    return clinical


def load_tmb():
    if not os.path.exists(TMB_PATH):
        raise FileNotFoundError(
            f"{TMB_PATH} not found - run downstream_analysis.py first "
            f"to generate per-sample TMB values."
        )
    return pd.read_csv(TMB_PATH)


def tmb_cox_model(merged):
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        print("lifelines not installed - skipping Cox model. pip install lifelines")
        return

    cox_df = merged[["time", "event", "TMB_mutations_per_Mb", "age_at_index"]].dropna()
    if len(cox_df) < 15:
        print(f"Only {len(cox_df)} cases with complete data - too few for a Cox model.")
        return

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="time", event_col="event")
    print("\nCox model: TMB (continuous) + age -> survival")
    print(cph.summary[["coef", "exp(coef)", "p"]])

    out_path = os.path.join(DOWNSTREAM_DIR, "cox_model_tmb.csv")
    cph.summary.to_csv(out_path)
    print(f"Saved -> {out_path}")


def tmb_quartile_km(merged):
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        print("lifelines not installed - skipping Kaplan-Meier plot. pip install lifelines")
        return

    df = merged.dropna(subset=["time", "event", "TMB_mutations_per_Mb"]).copy()
    if len(df) < 20:
        print(f"Only {len(df)} cases with complete data - too few to split into quartiles reliably.")
        return

    threshold = df["TMB_mutations_per_Mb"].quantile(0.75)
    df["tmb_group"] = df["TMB_mutations_per_Mb"].apply(
        lambda x: "TMB-high (top quartile)" if x >= threshold else "TMB-low/mid"
    )

    high = df[df["tmb_group"] == "TMB-high (top quartile)"]
    low = df[df["tmb_group"] == "TMB-low/mid"]

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))

    for group_df, label in [(high, f"TMB-high (n={len(high)})"), (low, f"TMB-low/mid (n={len(low)})")]:
        if len(group_df) > 0:
            kmf.fit(group_df["time"], group_df["event"], label=label)
            kmf.plot_survival_function()

    plt.title(f"Overall Survival by TMB Group (threshold = {threshold:.1f} mut/Mb, top quartile)")
    plt.xlabel("Days")
    plt.ylabel("Survival Probability")
    plt.tight_layout()

    plot_path = os.path.join(DOWNSTREAM_DIR, "km_survival_tmb_quartile.png")
    plt.savefig(plot_path)
    print(f"\nKaplan-Meier plot -> {plot_path}")

    if len(high) > 0 and len(low) > 0:
        result = logrank_test(
            high["time"], low["time"],
            event_observed_A=high["event"], event_observed_B=low["event"]
        )
        print(f"Log-rank test (TMB-high vs TMB-low/mid): p = {result.p_value:.4f}")


if __name__ == "__main__":
    tmb = load_tmb()
    clinical = load_clinical_with_survival()

    merged = tmb.merge(clinical, on="submitter_id", how="inner")
    n_with_time = merged["time"].notna().sum()
    print(f"Merged {len(merged)} samples with clinical data ({n_with_time} with usable survival time)")

    tmb_cox_model(merged)
    tmb_quartile_km(merged)
