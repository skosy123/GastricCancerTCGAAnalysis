"""
Stage-Stratified Survival Analysis
=====================================
Tumor stage is often the dominant driver of survival, and any gene-level
effect can be masked (or falsely appear/disappear) if stage isn't
accounted for. This script:

    1. Reports how many cases fall into each stage (so you can see whether
       stratification is even viable with your sample size).
    2. Fits a Cox model with gene mutation status + age + stage together.
    3. Plots Kaplan-Meier curves split by stage, if group sizes allow it.

Reads:
    maf_pipeline/results/combined_variants.csv
    maf_pipeline/results/gene_burden.csv
    clinical.cohort.<date>.json

Requires: pandas, lifelines
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
os.makedirs(DOWNSTREAM_DIR, exist_ok=True)

GENE = "TP53"                 # change to test a different gene's effect within stage
MIN_CASES_PER_STAGE = 5       # stages with fewer cases than this are grouped as "Other/rare"


def find_one(pattern):
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file found matching: {pattern}")
    return matches[0]


def link_barcode_to_case(tumor_sample_barcode):
    parts = str(tumor_sample_barcode).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else tumor_sample_barcode


def load_clinical_with_stage():
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
            "tumor_stage": primary.get("ajcc_pathologic_stage") or primary.get("tumor_stage"),
        })

    clinical = pd.DataFrame(rows)
    clinical["time"] = clinical["days_to_death"].fillna(clinical["days_to_last_follow_up"])
    clinical["event"] = clinical["days_to_death"].notna().astype(int)
    return clinical


def build_gene_mutation_column(gene):
    variants_path = os.path.join(RESULTS_DIR, "combined_variants.csv")
    if not os.path.exists(variants_path):
        raise FileNotFoundError(
            f"{variants_path} not found - run maf_pipeline.py first."
        )

    variants = pd.read_csv(variants_path, low_memory=False)
    samples = variants["Tumor_Sample_Barcode"].unique()
    mutated_samples = set(
        variants.loc[variants["Hugo_Symbol"] == gene, "Tumor_Sample_Barcode"].unique()
    )

    mat = pd.DataFrame({"Tumor_Sample_Barcode": samples})
    mat[gene] = mat["Tumor_Sample_Barcode"].isin(mutated_samples).astype(int)
    mat["submitter_id"] = mat["Tumor_Sample_Barcode"].apply(link_barcode_to_case)
    return mat[["submitter_id", gene]]


def simplify_stage(stage):
    """Collapse sub-stages (IIA, IIB -> Stage II) for larger, more usable groups."""
    if pd.isna(stage):
        return None
    stage = str(stage)
    for major in ["Stage IV", "Stage III", "Stage II", "Stage I"]:
        if stage.startswith(major):
            return major
    return stage


def report_stage_counts(clinical):
    clinical = clinical.copy()
    clinical["stage_group"] = clinical["tumor_stage"].apply(simplify_stage)
    counts = clinical["stage_group"].value_counts(dropna=False)

    print("Cases per stage group:")
    print(counts.to_string())

    viable_stages = [
        s for s in counts[counts >= MIN_CASES_PER_STAGE].index
        if pd.notna(s)
    ]
    if len(viable_stages) < 2:
        print(f"\nFewer than 2 stage groups (excluding missing/unknown stage) have "
              f">= {MIN_CASES_PER_STAGE} cases - stratified analysis will be "
              f"underpowered, but proceeding with what's available.")
    return clinical, viable_stages


def stage_adjusted_cox(clinical, mutation_col, gene):
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        print("lifelines not installed - skipping Cox model. pip install lifelines")
        return

    merged = clinical.merge(mutation_col, on="submitter_id", how="inner")
    merged = merged.dropna(subset=["time", "event", "age_at_index", "stage_group"])
    merged = merged[merged["stage_group"].notna()]

    if len(merged) < 20:
        print(f"\nOnly {len(merged)} cases with complete data (time, event, age, stage) "
              f"- too few for a stage-adjusted Cox model.")
        return

    cox_df = pd.get_dummies(
        merged[["time", "event", gene, "age_at_index", "stage_group"]],
        columns=["stage_group"], drop_first=True
    )

    try:
        from lifelines import CoxPHFitter
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="time", event_col="event")
        print(f"\nCox model: {gene} + age + stage -> survival (n={len(cox_df)})")
        print(cph.summary[["coef", "exp(coef)", "p"]])

        out_path = os.path.join(DOWNSTREAM_DIR, f"cox_model_{gene}_stage_adjusted.csv")
        cph.summary.to_csv(out_path)
        print(f"Saved -> {out_path}")
    except Exception as e:
        print(f"\nCox model failed to fit (often due to too few events per stage group): {e}")


def stage_km_plot(clinical, viable_stages):
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        print("lifelines not installed - skipping Kaplan-Meier plot. pip install lifelines")
        return

    df = clinical.dropna(subset=["time", "event"])
    df = df[df["stage_group"].isin(viable_stages)]

    if df["stage_group"].nunique() < 2:
        print("\nFewer than 2 usable stage groups with survival data - skipping stage KM plot.")
        return

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))

    for stage in sorted(df["stage_group"].unique()):
        group = df[df["stage_group"] == stage]
        kmf.fit(group["time"], group["event"], label=f"{stage} (n={len(group)})")
        kmf.plot_survival_function()

    plt.title("Overall Survival by Tumor Stage")
    plt.xlabel("Days")
    plt.ylabel("Survival Probability")
    plt.tight_layout()

    plot_path = os.path.join(DOWNSTREAM_DIR, "km_survival_by_stage.png")
    plt.savefig(plot_path)
    print(f"\nKaplan-Meier by stage -> {plot_path}")


if __name__ == "__main__":
    clinical = load_clinical_with_stage()
    clinical, viable_stages = report_stage_counts(clinical)

    mutation_col = build_gene_mutation_column(GENE)

    stage_adjusted_cox(clinical, mutation_col, GENE)
    stage_km_plot(clinical, viable_stages)