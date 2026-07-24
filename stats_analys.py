"""
Downstream Analysis: TCGA-STAD Mutation-Clinical Integration
===============================================================

"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

WORK_DIR = "maf_pipeline"
RESULTS_DIR = os.path.join(WORK_DIR, "results")
DOWNSTREAM_DIR = os.path.join(WORK_DIR, "downstream")
os.makedirs(DOWNSTREAM_DIR, exist_ok=True)

EXOME_SIZE_MB = 38.0  # standard approximation for WES capture size, used for TMB


def find_one(pattern):
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file found matching: {pattern}")
    return matches[0]


# ---------------------------------------------------------------------------
# Load prior results
# ---------------------------------------------------------------------------

def load_variants():
    path = os.path.join(RESULTS_DIR, "combined_variants.csv")
    variants = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(variants)} variants from {path}")
    return variants


def load_clinical_with_survival():
    """
    Re-reads the clinical JSON directly to pull survival fields
    (days_to_death, days_to_last_follow_up, vital_status) that the first
    script's loader didn't extract, since those are specifically needed
    here for Kaplan-Meier / Cox modeling.
    """
    path = find_one("clinical.cohort*.json")
    with open(path) as f:
        records = json.load(f)

    rows = []
    for rec in records:
        demographic = rec.get("demographic") or {}
        diagnoses = rec.get("diagnoses") or [{}]
        primary = diagnoses[0] if diagnoses else {}

        rows.append({
            "case_id": rec.get("case_id"),
            "submitter_id": rec.get("submitter_id"),
            "vital_status": demographic.get("vital_status"),
            "days_to_death": demographic.get("days_to_death"),
            "days_to_last_follow_up": demographic.get("days_to_last_follow_up")
                or primary.get("days_to_last_follow_up"),
            "age_at_index": demographic.get("age_at_index"),
            "primary_diagnosis": primary.get("primary_diagnosis"),
            "tumor_stage": primary.get("ajcc_pathologic_stage") or primary.get("tumor_stage"),
        })

    clinical = pd.DataFrame(rows)

    clinical["time"] = clinical["days_to_death"].fillna(clinical["days_to_last_follow_up"])
    clinical["event"] = clinical["days_to_death"].notna().astype(int)

    n_with_time = clinical["time"].notna().sum()
    print(f"Loaded clinical data: {len(clinical)} cases, {n_with_time} with usable survival time")
    return clinical


def link_barcode_to_case(tumor_sample_barcode):
    """TCGA aliquot barcodes are 'TCGA-XX-XXXX-...'; the first 3 fields
    (12 characters) identify the case/patient."""
    parts = str(tumor_sample_barcode).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else tumor_sample_barcode


# ---------------------------------------------------------------------------
# 1. Tumor mutational burden per sample
# ---------------------------------------------------------------------------

def compute_tmb(variants):
    per_sample = (
        variants
        .groupby("Tumor_Sample_Barcode")
        .size()
        .reset_index(name="variant_count")
    )
    per_sample["TMB_mutations_per_Mb"] = per_sample["variant_count"] / EXOME_SIZE_MB
    per_sample["submitter_id"] = per_sample["Tumor_Sample_Barcode"].apply(link_barcode_to_case)

    out_path = os.path.join(DOWNSTREAM_DIR, "tmb_per_sample.csv")
    per_sample.to_csv(out_path, index=False)
    print(f"\nTMB computed for {len(per_sample)} samples -> {out_path}")
    print(per_sample[["Tumor_Sample_Barcode", "TMB_mutations_per_Mb"]].describe())
    return per_sample


# ---------------------------------------------------------------------------
# 2. Gene-gene co-occurrence / mutual exclusivity
# ---------------------------------------------------------------------------

def gene_interaction_test(variants, top_genes):
    """
    For each pair of top genes, builds a per-sample mutated/not-mutated
    matrix and runs Fisher's exact test to flag significant co-occurrence
    (odds ratio > 1) or mutual exclusivity (odds ratio < 1).
    """
    samples = variants["Tumor_Sample_Barcode"].unique()
    mat = pd.DataFrame(0, index=samples, columns=top_genes)

    for gene in top_genes:
        mutated_samples = variants.loc[variants["Hugo_Symbol"] == gene, "Tumor_Sample_Barcode"].unique()
        mat.loc[mat.index.isin(mutated_samples), gene] = 1

    results = []
    for i, g1 in enumerate(top_genes):
        for g2 in top_genes[i + 1:]:
            table = pd.crosstab(mat[g1], mat[g2])
            table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            odds_ratio, p_value = fisher_exact(table)
            relationship = "co-occurring" if odds_ratio > 1 else "mutually exclusive"
            results.append({
                "gene_1": g1, "gene_2": g2,
                "odds_ratio": odds_ratio, "p_value": p_value,
                "relationship": relationship
            })

    results_df = pd.DataFrame(results).sort_values("p_value")
    out_path = os.path.join(DOWNSTREAM_DIR, "gene_interactions.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nGene interaction tests -> {out_path}")
    print(results_df.head(10))
    return results_df, mat


# ---------------------------------------------------------------------------
# 3. Predict vital status from top-gene mutation status
# ---------------------------------------------------------------------------

def predict_vital_status(mutation_matrix, clinical):
    mutation_matrix = mutation_matrix.copy()
    mutation_matrix["submitter_id"] = [link_barcode_to_case(b) for b in mutation_matrix.index]

    merged = mutation_matrix.merge(clinical[["submitter_id", "vital_status"]], on="submitter_id", how="inner")
    merged = merged.dropna(subset=["vital_status"])
    merged = merged[merged["vital_status"].isin(["Alive", "Dead"])]

    if len(merged) < 20:
        print(f"\nOnly {len(merged)} labeled cases available - too few for a reliable "
              f"classifier. Skipping predictive modeling.")
        return None

    feature_cols = [c for c in mutation_matrix.columns if c != "submitter_id"]
    X = merged[feature_cols]
    y = (merged["vital_status"] == "Dead").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\nLogistic regression: predicting vital status from {len(feature_cols)} gene mutation flags")
    print(f"Train/test split: {len(X_train)}/{len(X_test)} cases")
    if y_test.nunique() > 1:
        print(f"Test AUC: {roc_auc_score(y_test, probs):.3f}")
    print(classification_report(y_test, preds, zero_division=0))

    coef_table = pd.DataFrame({
        "gene": feature_cols,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", ascending=False)

    out_path = os.path.join(DOWNSTREAM_DIR, "vital_status_model_coefficients.csv")
    coef_table.to_csv(out_path, index=False)
    print(f"Model coefficients -> {out_path}")
    return model


# ---------------------------------------------------------------------------
# 4. Kaplan-Meier and Cox proportional hazards survival analysis
# ---------------------------------------------------------------------------

def survival_analysis(clinical, mutation_matrix, gene="TP53"):
    try:
        from lifelines import KaplanMeierFitter, CoxPHFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        print("\nlifelines not installed - skipping survival analysis. "
              "Install with: pip install lifelines")
        return

    import matplotlib.pyplot as plt

    mutation_matrix = mutation_matrix.copy()
    mutation_matrix["submitter_id"] = [link_barcode_to_case(b) for b in mutation_matrix.index]

    merged = clinical.merge(mutation_matrix[["submitter_id", gene]], on="submitter_id", how="inner")
    merged = merged.dropna(subset=["time", "event"])

    if len(merged) < 10:
        print(f"\nOnly {len(merged)} cases with survival data + {gene} status - skipping survival analysis.")
        return

    mutated = merged[merged[gene] == 1]
    wild_type = merged[merged[gene] == 0]

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))

    if len(mutated) > 0:
        kmf.fit(mutated["time"], mutated["event"], label=f"{gene} mutant (n={len(mutated)})")
        kmf.plot_survival_function()

    if len(wild_type) > 0:
        kmf.fit(wild_type["time"], wild_type["event"], label=f"{gene} wild-type (n={len(wild_type)})")
        kmf.plot_survival_function()

    plt.title(f"Overall Survival by {gene} Mutation Status")
    plt.xlabel("Days")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plot_path = os.path.join(DOWNSTREAM_DIR, f"km_survival_{gene}.png")
    plt.savefig(plot_path)
    print(f"\nKaplan-Meier plot -> {plot_path}")

    if len(mutated) > 0 and len(wild_type) > 0:
        result = logrank_test(
            mutated["time"], wild_type["time"],
            event_observed_A=mutated["event"], event_observed_B=wild_type["event"]
        )
        print(f"Log-rank test ({gene} mutant vs wild-type): p = {result.p_value:.4f}")

    # Cox model adjusting for age
    cox_df = merged[["time", "event", gene, "age_at_index"]].dropna()
    if len(cox_df) >= 15:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="time", event_col="event")
        print("\nCox proportional hazards model:")
        print(cph.summary[["coef", "exp(coef)", "p"]])
        cph.summary.to_csv(os.path.join(DOWNSTREAM_DIR, f"cox_model_{gene}.csv"))


# ---------------------------------------------------------------------------
# 5. [Optional / experimental] Neural ODE-based hazard model
# ---------------------------------------------------------------------------

def neural_ode_survival(clinical, mutation_matrix, gene="TP53"):
    """
    An exiperimental ODE-based ML model. Genuine fit here only because
    survival time is continuous - this parameterizes the hazard function
    h(t | x) with a small neural net and integrates the survival curve
    S(t) = exp(-integral of h) via an ODE solver, instead of assuming the
    proportional-hazards form Cox regression assumes.
    """
    try:
        import torch
        import torch.nn as nn
        from torchdiffeq import odeint
    except ImportError:
        print("\ntorch/torchdiffeq not installed - skipping Neural ODE section. "
              "Install with: pip install torch torchdiffeq")
        return

    mutation_matrix = mutation_matrix.copy()
    mutation_matrix["submitter_id"] = [link_barcode_to_case(b) for b in mutation_matrix.index]
    merged = clinical.merge(mutation_matrix[["submitter_id", gene]], on="submitter_id", how="inner")
    merged = merged.dropna(subset=["time", "event", "age_at_index"])

    if len(merged) < 30:
        print(f"\nOnly {len(merged)} usable cases - too few to attempt the Neural ODE "
              f"survival model meaningfully. Skipping.")
        return

    # Normalize covariates and time
    time = torch.tensor(merged["time"].values, dtype=torch.float32)
    time_norm = time / time.max()
    event = torch.tensor(merged["event"].values, dtype=torch.float32)
    covariates = torch.tensor(
        merged[[gene, "age_at_index"]].astype(float).values, dtype=torch.float32
    )
    covariates = (covariates - covariates.mean(0)) / (covariates.std(0) + 1e-6)

    class HazardODE(nn.Module):
        """dS/dt = -h(t, x) * S(t), h parameterized by a small MLP."""
        def __init__(self, n_covariates):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_covariates + 1, 16),
                nn.Tanh(),
                nn.Linear(16, 1),
                nn.Softplus(),  
            )
            self.x = None

        def set_covariates(self, x):
            self.x = x

        def forward(self, t, state):
            S = state
            t_expand = t.expand(self.x.shape[0], 1)
            hazard = self.net(torch.cat([t_expand, self.x], dim=1))
            dS_dt = -hazard.squeeze(-1) * S
            return dS_dt

    model = HazardODE(n_covariates=covariates.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    t_eval = torch.linspace(0, 1, 20)
    n_epochs = 200

    print(f"\nTraining Neural ODE hazard model on {len(merged)} cases ({n_epochs} epochs)...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        model.set_covariates(covariates)
        S0 = torch.ones(covariates.shape[0])
        S_traj = odeint(model, S0, t_eval)  # shape: [len(t_eval), n_samples]

        # Interpolate predicted survival probability at each patient's observed time
        idx = torch.clamp((time_norm * (len(t_eval) - 1)).long(), 0, len(t_eval) - 1)
        S_at_time = S_traj[idx, torch.arange(len(merged))]

        # Negative log-likelihood: events should have low S (near death), censored high S
        eps = 1e-6
        nll = -(event * torch.log(1 - S_at_time + eps) + (1 - event) * torch.log(S_at_time + eps))
        loss = nll.mean()

        loss.backward()
        optimizer.step()

        if epoch % 40 == 0:
            print(f"  epoch {epoch:3d}  loss {loss.item():.4f}")

    out_path = os.path.join(DOWNSTREAM_DIR, "neural_ode_hazard_model.pt")
    torch.save(model.state_dict(), out_path)
    print(f"Neural ODE model weights saved -> {out_path}")
    print("Reminder: this is exploratory given the small cohort size; validate "
          "against the Cox model results before drawing any conclusions.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    variants = load_variants()
    clinical = load_clinical_with_survival()

    tmb = compute_tmb(variants)

    top_genes = (
        pd.read_csv(os.path.join(RESULTS_DIR, "gene_burden.csv"))
        .head(15)["Hugo_Symbol"]
        .tolist()
    )
    interactions, mutation_matrix = gene_interaction_test(variants, top_genes)

    predict_vital_status(mutation_matrix, clinical)

    survival_analysis(clinical, mutation_matrix, gene="TP53")


    neural_ode_survival(clinical, mutation_matrix, gene="TP53")