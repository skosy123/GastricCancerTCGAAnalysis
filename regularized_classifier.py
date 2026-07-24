"""
Regularized Classifier + Univariate Gene Association Tests
===============================================================

    pip install pandas scipy scikit-learn
"""

import glob
import json
import os

import pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

WORK_DIR = "maf_pipeline"
RESULTS_DIR = os.path.join(WORK_DIR, "results")
DOWNSTREAM_DIR = os.path.join(WORK_DIR, "downstream")
os.makedirs(DOWNSTREAM_DIR, exist_ok=True)

TOP_N_GENES = 15


def find_one(pattern):
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file found matching: {pattern}")
    return matches[0]


def link_barcode_to_case(tumor_sample_barcode):
    parts = str(tumor_sample_barcode).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else tumor_sample_barcode


def load_clinical_vital_status():
    path = find_one("clinical.cohort*.json")
    with open(path) as f:
        records = json.load(f)

    rows = []
    for rec in records:
        demographic = rec.get("demographic") or {}
        rows.append({
            "submitter_id": rec.get("submitter_id"),
            "vital_status": demographic.get("vital_status"),
        })
    return pd.DataFrame(rows)


def build_mutation_matrix():
    variants_path = os.path.join(RESULTS_DIR, "combined_variants.csv")
    burden_path = os.path.join(RESULTS_DIR, "gene_burden.csv")

    if not (os.path.exists(variants_path) and os.path.exists(burden_path)):
        raise FileNotFoundError(
            "combined_variants.csv or gene_burden.csv not found under "
            "maf_pipeline/results - run maf_pipeline.py first."
        )

    variants = pd.read_csv(variants_path, low_memory=False)
    top_genes = pd.read_csv(burden_path).head(TOP_N_GENES)["Hugo_Symbol"].tolist()

    samples = variants["Tumor_Sample_Barcode"].unique()
    mat = pd.DataFrame(0, index=samples, columns=top_genes)

    for gene in top_genes:
        mutated = variants.loc[variants["Hugo_Symbol"] == gene, "Tumor_Sample_Barcode"].unique()
        mat.loc[mat.index.isin(mutated), gene] = 1

    mat["submitter_id"] = [link_barcode_to_case(b) for b in mat.index]
    return mat, top_genes


# ---------------------------------------------------------------------------
# 1. Univariate Fisher's exact tests: each gene vs vital status
# ---------------------------------------------------------------------------

def univariate_tests(mat, top_genes, clinical):
    merged = mat.merge(clinical, on="submitter_id", how="inner")
    merged = merged[merged["vital_status"].isin(["Alive", "Dead"])]

    results = []
    for gene in top_genes:
        table = pd.crosstab(merged[gene], merged["vital_status"])
        table = table.reindex(index=[0, 1], columns=["Alive", "Dead"], fill_value=0)
        odds_ratio, p_value = fisher_exact(table)
        n_mutated = int(merged[gene].sum())
        results.append({
            "gene": gene,
            "n_mutated": n_mutated,
            "n_wild_type": len(merged) - n_mutated,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
        })

    results_df = pd.DataFrame(results).sort_values("p_value")
    out_path = os.path.join(DOWNSTREAM_DIR, "univariate_gene_vital_status.csv")
    results_df.to_csv(out_path, index=False)

    print(f"\nUnivariate gene vs vital-status tests (n={len(merged)} labeled cases) -> {out_path}")
    print(results_df.to_string(index=False))
    return results_df


# ---------------------------------------------------------------------------
# 2. L1-regularized logistic regression with built-in CV for the penalty
# ---------------------------------------------------------------------------

def l1_logistic_regression(mat, top_genes, clinical):
    merged = mat.merge(clinical, on="submitter_id", how="inner")
    merged = merged[merged["vital_status"].isin(["Alive", "Dead"])]

    if len(merged) < 20:
        print(f"\nOnly {len(merged)} labeled cases - too few for regularized "
              f"logistic regression. Skipping.")
        return None

    X = merged[top_genes]
    y = (merged["vital_status"] == "Dead").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model = LogisticRegressionCV(
        penalty="l1", solver="liblinear", cv=5, max_iter=2000, random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\nL1-regularized logistic regression (best C = {model.C_[0]:.4f})")
    print(f"Train/test split: {len(X_train)}/{len(X_test)} cases")
    if y_test.nunique() > 1:
        print(f"Test AUC: {roc_auc_score(y_test, probs):.3f}")
    print(classification_report(y_test, preds, zero_division=0))

    coef_table = pd.DataFrame({
        "gene": top_genes,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False)

    n_nonzero = (coef_table["coefficient"] != 0).sum()
    print(f"\n{n_nonzero}/{len(top_genes)} gene coefficients retained (non-zero) after L1 penalty:")
    print(coef_table[coef_table["coefficient"] != 0].to_string(index=False))

    out_path = os.path.join(DOWNSTREAM_DIR, "l1_model_coefficients.csv")
    coef_table.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return model


if __name__ == "__main__":
    mat, top_genes = build_mutation_matrix()
    clinical = load_clinical_vital_status()

    univariate_tests(mat, top_genes, clinical)
    l1_logistic_regression(mat, top_genes, clinical)
