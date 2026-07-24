"""
TCGA-STAD Somatic Mutation (MAF) Pipeline

"""

import glob
import gzip
import json
import os
import tarfile

import pandas as pd
import requests
import matplotlib.pyplot as plt

WORK_DIR = "maf_pipeline"
DOWNLOAD_DIR = os.path.join(WORK_DIR, "downloads")
OUTPUT_DIR = os.path.join(WORK_DIR, "results")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_one(pattern):
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file found matching: {pattern}")
    if len(matches) > 1:
        print(f"Warning: multiple files match '{pattern}', using the first: {matches[0]}")
    return matches[0]


# ---------------------------------------------------------------------------
# Part 1: Narrow the full manifest down to just the somatic MAF files
# ---------------------------------------------------------------------------

def filter_manifest_to_maf(caller=None):
    """
    Reads the full GDC manifest and writes out a subset manifest containing
    only *.maf.gz files. Optionally restrict to a single caller
    (e.g. "MuTect2") if you don't want all callers' calls.
    """
    manifest_path = find_one("gdc_manifest.2026-07-13.140948.txt")
    manifest = pd.read_csv(manifest_path, sep="\t")

    maf_only = manifest[manifest["filename"].str.endswith(".maf.gz")]

    if caller:
        maf_only = maf_only[maf_only["filename"].str.contains(caller, case=False)]

    subset_path = os.path.join(WORK_DIR, "manifest_maf_only.txt")
    maf_only.to_csv(subset_path, sep="\t", index=False)

    total_gb = manifest["size"].sum() / 1e9
    subset_gb = maf_only["size"].sum() / 1e9
    print(f"Full manifest: {len(manifest)} files ({total_gb:.1f} GB)")
    print(f"MAF subset:    {len(maf_only)} files ({subset_gb:.4f} GB) -> {subset_path}")

    return subset_path


# ---------------------------------------------------------------------------
# Part 2: Download just those MAF files, directly via the GDC API
# ---------------------------------------------------------------------------

GDC_DATA_URL = "https://api.gdc.cancer.gov/data/{file_id}"
TOKEN_FILE = "gdc_token.txt"  # optional - only needed for controlled-access files


def load_auth_token():
    """
    If a GDC authentication token file is present, read it and return the
    headers dict to use for authenticated requests. Returns {} if no token
    file is found (open-access files don't need one).
    """
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            token = f.read().strip()
        print(f"Using GDC auth token from {TOKEN_FILE}")
        return {"X-Auth-Token": token}
    return {}


def download_maf_files(subset_manifest_path):
    """
    Downloads each file in the subset manifest straight from the GDC API
    using its file UUID. Open-access files download with no auth needed;
    controlled-access files need a valid token in gdc_token.txt or they'll
    be skipped (with a 403) and reported at the end - they are NOT
    something this script can or should work around.
    """
    manifest = pd.read_csv(subset_manifest_path, sep="\t")
    headers = load_auth_token()

    succeeded, forbidden, failed = [], [], []

    for i, row in manifest.iterrows():
        file_id = row["id"]
        filename = row["filename"]
        out_path = os.path.join(DOWNLOAD_DIR, filename)

        if os.path.exists(out_path):
            print(f"[{i+1}/{len(manifest)}] Already downloaded: {filename}")
            succeeded.append(filename)
            continue

        url = GDC_DATA_URL.format(file_id=file_id)
        print(f"[{i+1}/{len(manifest)}] Downloading {filename} ...")

        try:
            response = requests.get(url, headers=headers, stream=True, timeout=60)

            if response.status_code == 403:
                print(f"    -> 403 Forbidden: controlled-access file, skipping")
                forbidden.append(filename)
                continue

            response.raise_for_status()

            with open(out_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            succeeded.append(filename)

        except requests.exceptions.RequestException as e:
            print(f"    -> Failed: {e}")
            failed.append(filename)

    print(f"\nDownload summary: {len(succeeded)} succeeded, "
          f"{len(forbidden)} controlled-access (skipped), {len(failed)} other failures")

    if forbidden:
        print(f"\n{len(forbidden)} files need GDC controlled-access authorization "
              f"(see gdc_token.txt). These were NOT downloaded:")
        for f in forbidden[:10]:
            print(f"  - {f}")
        if len(forbidden) > 10:
            print(f"  ... and {len(forbidden) - 10} more")

    return succeeded


# ---------------------------------------------------------------------------
# Part 3: Load and combine all downloaded MAF files into one table
# ---------------------------------------------------------------------------

def load_and_combine_mafs():
    maf_files = glob.glob(os.path.join(DOWNLOAD_DIR, "**", "*.maf.gz"), recursive=True)

    if not maf_files:
        raise FileNotFoundError(
            f"No .maf.gz files found under {DOWNLOAD_DIR}. "
            f"If your download summary showed controlled-access files being skipped, "
            f"you'll need GDC authorization (see gdc_token.txt) before any variants "
            f"are available to analyze."
        )

    frames = []
    for maf_path in maf_files:
        with gzip.open(maf_path, "rt") as f:
            df = pd.read_csv(f, sep="\t", comment="#", low_memory=False)
        df["source_file"] = os.path.basename(maf_path)
        frames.append(df)
        print(f"Loaded {len(df)} variants from {os.path.basename(maf_path)}")

    combined = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, "combined_variants.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nCombined MAF table: {combined.shape[0]} variants, {combined.shape[1]} columns")
    print(f"Saved to: {out_path}")
    return combined


# ---------------------------------------------------------------------------
# Part 4: Load clinical data (from your GDC cohort export)
# ---------------------------------------------------------------------------

def load_clinical():
    """
    Loads clinical data regardless of which format GDC exported it in:
    a .tar.gz of TSVs, a plain .tsv, or a .json case-record export.
    """
    candidate = find_one("clinical.cohort*")

    if candidate.endswith(".json"):
        return load_clinical_json(candidate)

    extract_path = os.path.join(WORK_DIR, "clinical")
    os.makedirs(extract_path, exist_ok=True)

    if tarfile.is_tarfile(candidate):
        with tarfile.open(candidate) as tar:
            tar.extractall(extract_path)
    else:
        extract_path = os.path.dirname(candidate)

    clinical_file = find_one(os.path.join(extract_path, "**", "clinical.tsv"))
    clinical = pd.read_csv(clinical_file, sep="\t", na_values=["'--"])
    print(f"Loaded clinical data: {clinical.shape[0]} rows")
    return clinical


def load_clinical_json(path):
    """
    Flattens GDC's per-case clinical JSON export (nested demographic /
    diagnoses / follow_up dicts and lists) into one row per case with the
    fields most useful for downstream filtering, merging, and survival
    analysis.
    """
    with open(path) as f:
        records = json.load(f)

    rows = []
    for rec in records:
        project = rec.get("project") or {}
        demographic = rec.get("demographic") or {}
        diagnoses = rec.get("diagnoses") or [{}]
        primary = diagnoses[0] if diagnoses else {}
        follow_ups = rec.get("follow_ups") or [{}]

        vital_status = demographic.get("vital_status")
        days_to_death = demographic.get("days_to_death")

        # last follow-up can live in a few places depending on the export;
        # take whichever is present, preferring the most direct field
        days_to_last_follow_up = (
            primary.get("days_to_last_follow_up")
            or (follow_ups[0].get("days_to_follow_up") if follow_ups else None)
        )

        # survival_time / event are the two fields a time-to-event model
        # (Kaplan-Meier, Cox, or the ODE-based model below) actually needs
        if vital_status == "Dead" and days_to_death is not None:
            survival_time, event = days_to_death, 1
        elif days_to_last_follow_up is not None:
            survival_time, event = days_to_last_follow_up, 0
        else:
            survival_time, event = None, None

        rows.append({
            "case_id": rec.get("case_id"),
            "submitter_id": rec.get("submitter_id"),
            "project_id": project.get("project_id"),
            "disease_type": rec.get("disease_type"),
            "primary_site": rec.get("primary_site"),
            "gender": demographic.get("gender"),
            "race": demographic.get("race"),
            "ethnicity": demographic.get("ethnicity"),
            "vital_status": vital_status,
            "age_at_index": demographic.get("age_at_index"),
            "primary_diagnosis": primary.get("primary_diagnosis"),
            "tumor_stage": primary.get("ajcc_pathologic_stage") or primary.get("tumor_stage"),
            "morphology": primary.get("morphology"),
            "days_to_death": days_to_death,
            "days_to_last_follow_up": days_to_last_follow_up,
            "survival_time": survival_time,
            "event": event,  # 1 = death observed, 0 = censored
        })

    clinical = pd.DataFrame(rows)
    n_with_survival = clinical["survival_time"].notna().sum()
    print(f"Loaded clinical data (JSON): {clinical.shape[0]} cases "
          f"({n_with_survival} with usable survival_time/event)")
    return clinical


# ---------------------------------------------------------------------------
# Part 5: Filter to rare / pathogenic-impact variants
# ---------------------------------------------------------------------------

def filter_variants(variants):
    keep_cols_present = [c for c in ["Hugo_Symbol", "Variant_Classification", "IMPACT",
                                      "gnomAD_AF", "CLIN_SIG", "Tumor_Sample_Barcode"]
                          if c in variants.columns]
    print(f"Available annotation columns used for filtering: {keep_cols_present}")

    filtered = variants.copy()

    # Restrict to protein-changing, non-silent mutations
    coding_classes = [
        "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
        "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins", "Splice_Site"
    ]
    if "Variant_Classification" in filtered.columns:
        filtered = filtered[filtered["Variant_Classification"].isin(coding_classes)]

    # Drop common population variants if gnomAD frequency is annotated
    if "gnomAD_AF" in filtered.columns:
        filtered = filtered[(filtered["gnomAD_AF"].isna()) | (filtered["gnomAD_AF"] < 0.01)]

    print(f"Filtered to {len(filtered)} coding, rare variants (from {len(variants)} total)")
    return filtered


# ---------------------------------------------------------------------------
# Part 6: Gene-level mutation burden
# ---------------------------------------------------------------------------

def gene_burden(filtered):
    burden = (
        filtered
        .groupby("Hugo_Symbol")
        .size()
        .reset_index(name="Variant_Count")
        .sort_values("Variant_Count", ascending=False)
    )

    out_path = os.path.join(OUTPUT_DIR, "gene_burden.csv")
    burden.to_csv(out_path, index=False)
    print(f"Gene burden table saved to: {out_path}")
    print(burden.head(20))
    return burden


# ---------------------------------------------------------------------------
# Part 6b: Per-sample tumor mutational burden (TMB)
# ---------------------------------------------------------------------------

def compute_tmb(filtered, exome_size_mb=38.0):
    """
    Counts filtered (coding, rare) variants per sample and expresses that
    as mutations/Mb, the standard TMB unit. exome_size_mb defaults to the
    commonly used ~38 Mb WXS coding footprint; adjust if you know the
    actual capture size used for these samples.
    """
    tmb = (
        filtered
        .groupby("Tumor_Sample_Barcode")
        .size()
        .reset_index(name="Variant_Count")
    )
    tmb["TMB_per_Mb"] = tmb["Variant_Count"] / exome_size_mb
    tmb = tmb.sort_values("TMB_per_Mb", ascending=False)

    out_path = os.path.join(OUTPUT_DIR, "tmb_per_sample.csv")
    tmb.to_csv(out_path, index=False)
    print(f"TMB per sample saved to: {out_path}")
    return tmb


# ---------------------------------------------------------------------------
# Part 7: Flag known gastric-cancer-relevant genes
# ---------------------------------------------------------------------------

def check_known_genes(burden):
    known_genes = [
        "TP53", "CDH1", "ARID1A", "PIK3CA", "ERBB2",
        "KRAS", "SMAD4", "RHOA", "CTNNB1", "APC"
    ]
    candidate = burden[burden["Hugo_Symbol"].isin(known_genes)]
    print("\nKnown gastric-cancer-related genes found in this cohort:")
    print(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Part 8: Plot top mutated genes
# ---------------------------------------------------------------------------

def plot_top_genes(burden, top_n=20):
    top = burden.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.bar(top["Hugo_Symbol"], top["Variant_Count"])
    plt.xticks(rotation=90)
    plt.ylabel("Number of Coding, Rare Variants")
    plt.title("Most Frequently Mutated Genes")
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "top_mutated_genes.png")
    plt.savefig(plot_path)
    print(f"\nSaved plot: {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    subset_manifest = filter_manifest_to_maf()
    download_maf_files(subset_manifest)

    variants = load_and_combine_mafs()
    clinical = load_clinical()
    clinical.to_csv(os.path.join(OUTPUT_DIR, "clinical_data.csv"), index=False)
    print(f"Clinical data saved to: {os.path.join(OUTPUT_DIR, 'clinical_data.csv')}")

    filtered = filter_variants(variants)
    filtered.to_csv(os.path.join(OUTPUT_DIR, "filtered_variants.csv"), index=False)

    burden = gene_burden(filtered)
    tmb = compute_tmb(filtered)
    check_known_genes(burden)
    plot_top_genes(burden)


