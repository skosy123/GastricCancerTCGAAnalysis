"""
FDR Correction for Gene Interaction Tests
===========================================

pip install pandas statsmodels
"""

import os
import pandas as pd
from statsmodels.stats.multitest import multipletests

WORK_DIR = "maf_pipeline"
DOWNSTREAM_DIR = os.path.join(WORK_DIR, "downstream")
INPUT_PATH = os.path.join(DOWNSTREAM_DIR, "gene_interactions.csv")
OUTPUT_PATH = os.path.join(DOWNSTREAM_DIR, "gene_interactions_fdr.csv")

ALPHA = 0.05


def apply_fdr_correction():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"{INPUT_PATH} not found - run downstream_analysis.py first "
            f"to generate the gene interaction test results."
        )

    results = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(results)} pairwise gene tests from {INPUT_PATH}")

    reject, q_values, _, _ = multipletests(results["p_value"], alpha=ALPHA, method="fdr_bh")
    results["q_value"] = q_values
    results["significant_fdr"] = reject

    results = results.sort_values("q_value")
    results.to_csv(OUTPUT_PATH, index=False)

    n_sig_raw = (results["p_value"] < ALPHA).sum()
    n_sig_fdr = results["significant_fdr"].sum()

    print(f"\nBefore correction: {n_sig_raw}/{len(results)} pairs significant at raw p < {ALPHA}")
    print(f"After BH-FDR correction: {n_sig_fdr}/{len(results)} pairs significant at q < {ALPHA}")
    print(f"\nSaved -> {OUTPUT_PATH}")

    if n_sig_fdr > 0:
        print("\nGene pairs surviving FDR correction:")
        print(results[results["significant_fdr"]][
            ["gene_1", "gene_2", "odds_ratio", "p_value", "q_value", "relationship"]
        ].to_string(index=False))
    else:
        print("\nNo pairs survive FDR correction - the raw hits are likely noise from "
              "multiple testing rather than real biological interactions in this cohort.")

    return results


if __name__ == "__main__":
    apply_fdr_correction()
