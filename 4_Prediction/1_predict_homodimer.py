#!/usr/bin/env python3
"""
predict_homodimer.py
=======================
Improve homodimer identification (target metric: TPR @ 1% FPR) over the
logistic-regression approach of Narrowe Danielsson & Elofsson (2025).

It builds the following scorers and compares them under the evaluation protocol selected
on the command line (see EVALUATION PROTOCOLS below):

  baselines (reproduce the paper)
    * af_pred                       : single feature (max ranking confidence,
                                      0.8*ipTM + 0.2*pTM)                       ~0.18
    * narrowe_logreg_homology       : 8-feature logistic regression (Table 3 L) ~0.42 ± 0.08
    * narrowe_logreg_nohomology     : 5-feature logistic regression (Table 3 R) ~0.28 ± 0.07
    * narrowe_logreg_homology_all_featr   : same LR but ALL features (incl. homology)
    * narrowe_logreg_nohomology_all_featr : same LR but ALL features except homology

  new models (this script)
    * MODEL 3  random_forest   : Random Forest, pooled-OOF grid search selected 3 ways
                                 (best AUPR / TPR@5% / TPR@1%), seed-averaged.
    * MODEL 4  xgboost         : XGBoost, monotonic + native NaN, pooled-OOF grid search
                                 selected 3 ways, seed-averaged (needs `pip install xgboost`).

EVALUATION PROTOCOLS (two modes)
  (A) DEFAULT - temporal split: tune by pooled-OOF stratified CV inside the PRE-cutoff
      set ("before"), then test ONCE on the POST-cutoff set ("after"). "after" was never
      seen by AlphaFold2.3, so it is the honest, real-world estimate. The 3-way RF/XGB
      selection and the validation/test tables come from this mode. `--nested-cv` adds a
      selection-unbiased nested-CV estimate (on "before").
      NOTE: because the models use AlphaFold confidence as features, and AF memorised the
      pre-cutoff structures, "before" features are inflated; this mode is kept mainly for
      the AF-memorisation analysis (before-vs-after gap), not as the model headline.

  (B) --after-only-nested - the recommended model evaluation: nested CV on the AFTER-set
      ONLY (honest AF features in both train and test, matching Narrowe's all-post-cutoff
      design). All-feature LR (tunes C), Random Forest, and XGBoost are each nested-CV'd
      (inner folds select the config by AUPR; outer folds grade on pooled OOF). af_pred and
      the Narrowe pruned LR are added as reference rows.

  Every reported number carries a stratified-bootstrap 95% CI, because TPR@1%FPR pivots on
  only ~6 negatives and is very noisy.

INPUTS
  --features-before / --features-after : the two .tsv feature tables
                                         (before-args optional with --after-only-nested)
  --labels-before / --labels-after     : ids_labels.csv  (columns: id, [chain,] label)

Usage examples:
  # (A) temporal split (optionally + nested-CV cross-check on 'before')
  python predict_homodimer.py \
      --features-before features_before.tsv --features-after features_after.tsv \
      --labels-before ids_labels_before.csv --labels-after ids_labels_after.csv \
      --outdir results [--nested-cv-val]

  # (B) after-only nested CV (the recommended model evaluation)
  python predict_homodimer.py \
      --features-after features_after.tsv --labels-after ids_labels_after.csv \
      --after-only-nested --outer-k 5 --inner-k 5 --outdir results_after_only

Requires: scikit-learn>=1.8, pandas, numpy, scipy, matplotlib.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:                                  # optional: SOTA gradient booster (MODEL 4)
    import xgboost as xgb
except Exception:
    xgb = None

RNG = 42
np.random.seed(RNG)

# ============================================================================ #
# 1. FEATURE-GROUP DEFINITIONS  (edit here if your column names differ)
# ============================================================================ #
AF_CONF = [
    "max_iptm", "min_iptm", "avg_iptm",
    "max_ptm", "min_ptm", "avg_ptm",
    "max_rc", "min_rc", "avg_rc",
]
HOMOLOGY_NUM = (
    [f"multimer_frac_tm{t/10:.1f}" for t in range(10)]
    + [f"hm_frac_tm{t/10:.1f}" for t in range(10)]
    + ["highest_tm_all_hits", "highest_tm_multimers", "highest_tm_homomultimers"]
)
HOMOLOGY_CAT = ["stoich_all_hits", "stoich_multimers"]
HOMOLOG_IDS = [                                             # PDB-id strings -> drop
    "highest_match_all_hits", "highest_match_multimers", "highest_match_homomultimers",
]
SPOC = [
    "num_contacts_with_max_n_models", "num_unique_contacts",
    "mean_contacts_across_predictions", "min_contacts_across_predictions",
    "best_num_residue_contacts", "best_if_residues",
    "best_plddt_max", "best_pae_min", "best_contact_score_max",
]
SASA = [
    "buried_apolar_area_mean", "buried_polar_area_mean", "total_interaction_area_mean",
    "fraction_buried_apolar_area_mean", "fraction_buried_polar_area_mean",
    "buried_apolar_area_min", "buried_polar_area_min", "total_interaction_area_min",
    "fraction_buried_apolar_area_min", "fraction_buried_polar_area_min",
    "buried_apolar_area_max", "buried_polar_area_max", "total_interaction_area_max",
    "fraction_buried_apolar_area_max", "fraction_buried_polar_area_max",
]
CONSENSUS = ["structural_consensus_mean", "structural_consensus_min", "structural_consensus_max"]
IPSAE = ["min_ipsae", "max_ipsae", "avg_ipsae"]   # dropped in preprocess(); not used as features

NUMERIC = AF_CONF + HOMOLOGY_NUM + SPOC + SASA + CONSENSUS

# tool-groups whose total failure is itself informative -> add a missingness flag
HOMOLOGY = HOMOLOGY_NUM + HOMOLOGY_CAT
# type-specific homology cols: legitimately NaN when hits exist but none are of that type
HOM_TYPED = ["highest_tm_multimers", "highest_tm_homomultimers", "stoich_multimers"]
# core homology cols: must be present whenever ANY homolog hit exists
HOM_CORE = [c for c in HOMOLOGY if c not in HOM_TYPED]
STOICH_CATEGORIES = ["monomer", "homomultimer", "heteromultimer", "none"]

# ---- monotonic priors for the gradient-boosted model -----------------------
# +1 : larger value -> MORE likely a homodimer ; -1 : larger -> LESS likely ; 0 : no prior
def build_monotonic_map() -> dict:
    m = {}
    for c in (
        [f"hm_frac_tm{t/10:.1f}" for t in range(10)]
        + [f"multimer_frac_tm{t/10:.1f}" for t in range(10)]
        + ["highest_tm_homomultimers", "highest_tm_multimers"]
        + AF_CONF
        + ["num_contacts_with_max_n_models", "num_unique_contacts",
           "mean_contacts_across_predictions", "min_contacts_across_predictions",
           "best_num_residue_contacts", "best_if_residues",
           "best_plddt_max", "best_contact_score_max"]
        + CONSENSUS
        + ["buried_apolar_area_mean", "buried_polar_area_mean", "total_interaction_area_mean",
           "buried_apolar_area_min", "buried_polar_area_min", "total_interaction_area_min",
           "buried_apolar_area_max", "buried_polar_area_max", "total_interaction_area_max"]
    ):
        m[c] = 1
    m["best_pae_min"] = -1                       # lower predicted-aligned-error is better
    # everything else (highest_tm_all_hits, fraction_* ratios, dummies, flags) -> 0 (omitted)
    return m


# ============================================================================ #
# 2. DATA LOADING & PREPROCESSING
# ============================================================================ #
def _normalize_features(df: pd.DataFrame, source: str = "features") -> pd.DataFrame:
    """Shared feature-frame normalization: drop the blank/unnamed columns produced by
    stray tabs, require an 'ID' column, and cast it to str. Idempotent, so it is safe to
    apply again to a frame that already passed through _read_features()."""
    keep = [c for c in df.columns if str(c).strip() and not str(c).startswith("Unnamed")]
    out = df[keep].copy()
    if "ID" not in out.columns:
        raise ValueError(f"{source}: no 'ID' column found.")
    out["ID"] = out["ID"].astype(str)
    return out


def _read_features(path: Path) -> pd.DataFrame:
    return _normalize_features(pd.read_csv(path, sep="\t"), source=str(path))


def _normalize_key(s: pd.Series) -> pd.Series:
    """Join key: first 4 chars of the PDB id, stripped and lower-cased.
    Feature IDs like '7B1K1' -> '7b1k' to match ids_labels ('7acw')."""
    return s.astype(str).str.strip().str[:4].str.lower()


def load_df(features_df: pd.DataFrame, labels: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Attach labels to an in-memory features DataFrame (no clustering).
    'split_name': to distinguish data before/after AF2.3 training cutoff.
    """
    feats = _normalize_features(features_df, source=f"{split_name} features")
    feats["_key"] = _normalize_key(feats["ID"])
    lab = labels.copy()
    lab["_key"] = _normalize_key(lab["id"])
    # keep 'stoichiometry' when the labels file provides it, for the dataset statistics
    lab_cols = ["_key", "label"] + (["stoichiometry"] if "stoichiometry" in lab.columns else [])
    lab = lab[lab_cols].drop_duplicates("_key")
    merged = feats.merge(lab, on="_key", how="inner")
    if len(merged) < len(feats):
        warnings.warn(f"[{split_name}] {len(feats) - len(merged)} rows unmatched to labels; dropped.")
    merged["split"] = split_name
    return merged


def preprocess(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Row/column cleaning applied to a labelled split BEFORE build_design():
      (1)     drop IPSAE and HOMOLOG_IDS columns
      (2)     drop rows with ANY NaN in SPOC/SASA/CONSENSUS (tool failed -> unscoreable)
      (3)     homology missingness:
                - partial in CORE cols  -> BROKEN extraction: print error + drop
                - all-NaN (no homolog)  -> fill numeric with 0, set no_homolog flag = 1
                - partial in TYPED cols -> fill numeric with 0 (benign "no hit of that type")
              (categorical stoich NaNs become 'none' later in build_design)
    
    'split_name': to distinguish data before/after AF2.3 training cutoff.
    """
    df = df.copy()

    # (1) drop IPSAE and HOMOLOG_IDS columns (+ any column containing 'ipsae')
    drop_cols = [c for c in df.columns
                 if c in HOMOLOG_IDS or c in IPSAE or "ipsae" in str(c).lower()]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    # (2) drop rows with ANY NaN in SPOC / SASA / CONSENSUS
    struct = [c for c in (SPOC + SASA + CONSENSUS) if c in df.columns]
    n0 = len(df)
    df = df.dropna(subset=struct).copy() if struct else df
    print(f"[{split_name}] dropped {n0 - len(df)} rows with missing SPOC/SASA/CONSENSUS "
          f"-> {len(df)} rows remain")

    # (3) homology missingness
    hom = [c for c in HOMOLOGY if c in df.columns]
    core = [c for c in HOM_CORE if c in df.columns]
    hom_num = [c for c in HOMOLOGY_NUM if c in df.columns]

    na_all = df[hom].isna().all(axis=1)                          # no homolog at all
    core_na = df[core].isna().any(axis=1) if core else pd.Series(False, index=df.index)
    broken = ~na_all & core_na      # partial in CORE -> broken

    if broken.any():
        bad = df.loc[broken, core].isna().sum()
        ids = df.loc[broken, "ID"].head(8).tolist() if "ID" in df.columns else []
        print(f"  ERROR [{split_name}]: {int(broken.sum())} rows have NaNs in CORE homology "
              f"columns (extraction half-failed) -> dropping them.\n"
              f"    core cols with NaNs: {bad[bad > 0].to_dict()}\n"
              f"    example IDs: {ids}\n"
              f"    suggestion: re-run Foldseek/MMseqs for these IDs; do NOT zero-fill.")
        df = df.loc[~broken].copy()
        na_all = na_all.loc[df.index]

    # no_homolog flag set
    df["no_homolog"] = na_all.astype("float64")
    # fill numeric homology NaNs with 0 (covers both no_homolog and partial_typed rows)
    if hom_num:
        df[hom_num] = df[hom_num].fillna(0.0)
    print(f"  [{split_name}] no_homolog rows: {int(df['no_homolog'].sum())}")
    return df



def build_design(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model-ready design matrix: numeric (NaN kept) + homolog missingness flags +
    one-hot stoichiometry, in a stable column order."""
    present_num = [c for c in NUMERIC if c in df.columns]
    missing_num = [c for c in NUMERIC if c not in df.columns]
    if missing_num:
        warnings.warn(f"{len(missing_num)} expected numeric features absent "
                      f"(e.g. {missing_num[:3]}). They are skipped.")
    X = df[present_num].apply(pd.to_numeric, errors="coerce").astype("float64")

    # carry the no_homolog flag produced by preprocess() (homology NaNs are filled)
    extra = [c for c in ["no_homolog"] if c in df.columns]
    X_flags = df[extra].astype("float64") if extra else pd.DataFrame(index=df.index)

    # one-hot the categorical stoichiometry columns with fixed categories (no leakage)
    STOICH_CATEGORIES_BY_COL = {
    "stoich_all_hits":  STOICH_CATEGORIES,
    "stoich_multimers": ["homomultimer", "heteromultimer", "none"],
    }

    def stoich_categories(col: str) -> list:
        """Allowed stoichiometry levels for a categorical homology column."""
        return STOICH_CATEGORIES_BY_COL.get(col, STOICH_CATEGORIES)
        
    cat_present = [c for c in HOMOLOGY_CAT if c in df.columns]
    if cat_present:
        cat = df[cat_present].astype("object")
        cat = cat.where(cat.notna(), "none")
        cat = cat.apply(lambda s: s.str.lower().where(
            s.str.lower().isin(stoich_categories(s.name)), "none"))
        ohe = OneHotEncoder(categories=[stoich_categories(c) for c in cat_present],
                            handle_unknown="ignore", sparse_output=False)
        arr = ohe.fit_transform(cat)
        names = ohe.get_feature_names_out(cat_present)
        X_cat = pd.DataFrame(arr, columns=names, index=df.index).astype("float64")
    else:
        X_cat = pd.DataFrame(index=df.index)

    design = pd.concat([X, X_flags, X_cat], axis=1)
    design.columns = [str(c) for c in design.columns]
    return design


# ============================================================================ #
# 3. METRICS  (TPR@FPR, partial AUC, MCC, stratified bootstrap CI)
# ============================================================================ #
def tpr_at_fpr(y_true, y_score, target_fpr=0.01) -> float:
    """TPR at the most permissive threshold whose FPR does not exceed target_fpr."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= target_fpr + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def partial_auc(y_true, y_score, max_fpr=0.05) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))


def mcc_at_best_threshold(y_true, y_score) -> float:
    """Matthews correlation coefficient at the threshold that maximises it.

    MCC needs a hard 0/1 cut-off, but the scorers here live on different scales: AlphaFold
    ranking confidence is calibrated 0-1, whereas RF/XGB/LR emit model probabilities that
    may never reach a fixed cut like 0.8. A single fixed threshold would therefore punish
    models for their score scale rather than their discrimination, so each model is scored
    at its own optimum.

    Computed in closed form from the ROC curve, so every candidate threshold is evaluated.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype="float64")
    ok = ~np.isnan(y_score)
    y_true, y_score = y_true[ok], y_score[ok]
    P = float((y_true == 1).sum())
    N = float((y_true == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    tp, fp = tpr * P, fpr * N
    fn, tn = P - tp, N - fp
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        mcc = np.where(denom > 0, (tp * tn - fp * fn) / denom, 0.0)
    return float(np.nanmax(mcc))


def all_metrics(y_true, y_score) -> dict:
    return {
        "AUPR": float(average_precision_score(y_true, y_score)),
        "pAUC@5%": partial_auc(y_true, y_score, 0.05),
        "MCC": mcc_at_best_threshold(y_true, y_score),
        "TPR@5%FPR": tpr_at_fpr(y_true, y_score, 0.05),
        "TPR@1%FPR": tpr_at_fpr(y_true, y_score, 0.01),
    }


def stratified_bootstrap_ci(y_true, y_score, metric_fn, n_boot=2000, alpha=0.05, seed=RNG):
    """Resample positives and negatives separately (keeps class counts fixed)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        v = metric_fn(y_true[idx], y_score[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    vals = np.array(vals)
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# ============================================================================ #
# 4. MODELS
# ============================================================================ #
def _logreg_pipe():
    """Standard logistic-regression pipeline (median impute + scale + balanced L2)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RNG)),
    ])


def homology_design_columns(design_columns):
    """Columns in the design matrix that are derived from homology (Foldseek) info; after one-hot encoding."""
    pref = ("hm_frac_tm", "multimer_frac_tm", "highest_tm_", "stoich_all_hits", "stoich_multimers")
    return [c for c in design_columns if c.startswith(pref) or c == "no_homolog"]


def make_narrowe_logreg(with_homology: bool):
    """Reproduction of the paper's logistic regression (Table 3, hand-pruned features)."""
    if with_homology:
        feats = ["hm_frac_tm0.8", "hm_frac_tm0.9", "multimer_frac_tm0.6", "multimer_frac_tm0.8",
                 "total_interaction_area_mean", "best_plddt_max", "best_pae_min",
                 "structural_consensus_mean"]
    else:
        feats = ["num_unique_contacts", "best_plddt_max", "best_pae_min", "max_iptm", "avg_iptm"]
    return feats, _logreg_pipe()


def make_logreg_all(tune=False):
    """All-feature logistic regression. For nested CV it tunes the L2 strength C."""
    pipe = _logreg_pipe()
    if not tune:
        return pipe, None
    return pipe, {"clf__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}


def make_model3_rf(tune=False):
    """MODEL 3 : Random Forest."""
    pipe = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=1, random_state=RNG)),
    ])
    if not tune:
        return pipe, None
    grid = {
        "clf__max_features": [0.1, 0.2, 0.3],
        "clf__min_samples_leaf": [2, 5, 10],
    }
    return pipe, grid


def make_model4_xgb(design_columns, scale_pos_weight=1.0, seed=RNG):
    """MODEL 4 : XGBoost. Native NaN handling + monotonic constraints.
    Returns None if xgboost is not installed."""
    if xgb is None:
        return None
    mono = build_monotonic_map()
    constraints = tuple(int(mono.get(c, 0)) for c in design_columns)  # aligned to X columns
    return xgb.XGBClassifier(
        n_estimators=600, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=5,
        monotone_constraints=constraints, scale_pos_weight=scale_pos_weight,
        tree_method="hist", eval_metric="logloss", n_jobs=1, random_state=seed,
    )


def xgb_grid():
    """Modest XGBoost tuning grid."""
    return {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1],
        "min_child_weight": [1, 5],
        "reg_lambda": [1.0, 10.0],
    }


def _score(fitted, X) -> np.ndarray:
    if hasattr(fitted, "predict_proba"):
        return fitted.predict_proba(X)[:, 1]
    return fitted.decision_function(X)


def nested_cv_oof(base, grid, X, y, outer_k=5, inner_k=5, seed=RNG):
    """Nested cross-validation, returning pooled OUTER out-of-fold predictions.

    Outer loop grades; inner loop selects. For each outer fold: run a flat inner-CV grid
    search on the development folds, pick the config with the best pooled-OOF performer, 
    train it on all dev folds, and predict the held-out outer fold.
    """
    outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=seed)
    inner = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan, dtype="float64")
    ls_params = list(ParameterGrid(grid))
    for train, test in outer.split(np.zeros(len(y)), y):
        Xdev, ydev = X.iloc[train], y[train]
        best_score, best_params = -np.inf, None
        for params in ls_params:                       # inner selection on dev only
            est = clone(base).set_params(**params)
            score = cross_val_predict(est, Xdev, ydev, cv=inner,
                                   method="predict_proba", n_jobs=4)[:, 1]
            s = tpr_at_fpr(ydev, score, 0.01)  # select by TPR at 1% FPR
            if s > best_score:
                best_score, best_params = s, params
        winner = clone(base).set_params(**best_params).fit(Xdev, ydev)
        oof[test] = winner.predict_proba(X.iloc[test])[:, 1]
    return oof

# ---- seed averaging (de-noise stochastic models) ------------------
def reseed(est, seed):
    """Clone an estimator and set EVERY random_state (incl. nested) to `seed`."""
    e = clone(est)
    params = {k: seed for k in e.get_params(deep=True)
              if k == "random_state" or k.endswith("__random_state")}
    if params:
        e.set_params(**params)
    return e


def seed_avg_scores(est, Xb, yb, Xa, seeds):
    """Fit `est` under several seeds and average the after-set scores (bagging over seeds)."""
    return np.mean([_score(reseed(est, s).fit(Xb, yb), Xa) for s in seeds], axis=0)


# ============================================================================ #
# 5. EVALUATION  (A: before->after temporal split + optional nested CV;
#                 B: after-only nested CV)
# ============================================================================ #
def _jsonable(d: dict) -> dict:
    """Keep only JSON-serialisable scalar/list params (tuples -> lists)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(
                isinstance(x, (int, float, str, bool)) for x in v):
            out[k] = list(v)
    return out


def evaluate_all(before: pd.DataFrame, after: pd.DataFrame, outdir: Path,
                 n_boot=2000, n_seeds=5, cv_folds=5,
                 with_nested=False, outer_k=5, inner_k=5):
    Xb_full = build_design(before)
    Xa_full = build_design(after).reindex(columns=Xb_full.columns)  # align columns
    yb = before["label"].astype(int).values
    ya = after["label"].astype(int).values
    cols = list(Xb_full.columns)
    seeds = [RNG + i for i in range(max(1, n_seeds))]
    spw = float((yb == 0).sum()) / max(1, int((yb == 1).sum()))     # xgb scale_pos_weight
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RNG)

    # AF ranking-confidence as a raw column on the BEFORE rows (no model -> its own OOF)
    af_b = pd.to_numeric(before.get("max_rc"), errors="coerce")
    if af_b.isna().all():
        af_b = 0.8 * pd.to_numeric(before.get("max_iptm"), errors="coerce") \
               + 0.2 * pd.to_numeric(before.get("max_ptm"), errors="coerce")
    af_b = af_b.values

    test_results, test_scores = {}, {}
    val_results, val_scores, tuned_params = {}, {}, {}
    step = {"i": 0}

    def hdr(title):
        """step-counter / section-header printer; prints a numbered heading each time a new model or stage runs."""
        step["i"] += 1
        print(f"\n[{step['i']}] {title}")

    def oof(est, Xb):
        """Pooled out-of-fold positive-class probabilities over the before-set."""
        return cross_val_predict(clone(est), Xb, yb, cv=skf,
                                 method="predict_proba", n_jobs=4)[:, 1]

    def _ci(y, s):
        s = np.asarray(s, dtype="float64")
        lo1, hi1 = stratified_bootstrap_ci(y, s, lambda yt, ys: tpr_at_fpr(yt, ys, 0.01), n_boot=n_boot)
        lo5, hi5 = stratified_bootstrap_ci(y, s, lambda yt, ys: tpr_at_fpr(yt, ys, 0.05), n_boot=n_boot)
        return (lo1, hi1), (lo5, hi5)

    def _rec(name, score, y_true, results_dict, scores_dict=None):
        score = np.asarray(score, dtype="float64")
        if scores_dict is not None:
            scores_dict[name] = score
        m = all_metrics(y_true, score)
        m["TPR@1%FPR_CI"], m["TPR@5%FPR_CI"] = _ci(y_true, score)
        results_dict[name] = m
        print(f"  {name:30s}  AUPR={m['AUPR']:.3f}  pAUC@5%={m['pAUC@5%']:.3f}  "
            f"MCC={m['MCC']:.3f}  TPR@5%={m['TPR@5%FPR']:.3f} "
            f"[{m['TPR@5%FPR_CI'][0]:.3f},{m['TPR@5%FPR_CI'][1]:.3f}]  "
            f"TPR@1%={m['TPR@1%FPR']:.3f} [{m['TPR@1%FPR_CI'][0]:.3f},{m['TPR@1%FPR_CI'][1]:.3f}]")

    # ===================================================================== #
    # PHASE 1 - VALIDATION: stratified k-fold CV on the before-set. Every
    #   metric is computed on the POOLED out-of-fold predictions one training
    #   set, with a stratified-bootstrap CI. The test-set
    #   is NOT touched here. RF hyper-parameters are selected three ways.
    # ===================================================================== #
    print(f"\n################  VALIDATION ({cv_folds}-fold pooled out-of-fold CV on 'before')"
          f"  ################")

    _rec("af_pred (max rank-conf)", af_b, yb, val_results, val_scores)

    for tag, with_h in [("narrowe_logreg_homology", True), ("narrowe_logreg_nohomology", False)]:
        feats, pipe = make_narrowe_logreg(with_h)
        feats = [f for f in feats if f in cols]
        _rec(tag, oof(pipe, Xb_full[feats]), yb, val_results, val_scores)

    hom_cols = homology_design_columns(cols)
    allfeat = {
        "narrowe_logreg_homology_all_featr": cols,
        "narrowe_logreg_nohomology_all_featr": [c for c in cols if c not in hom_cols],
    }
    for tag, feats in allfeat.items():
        _rec(tag, oof(_logreg_pipe(), Xb_full[feats]), yb, val_results, val_scores)

    # MODEL 3 : Random Forest -- pooled-OOF grid search, selected 3 ways
    base3, grid3 = make_model3_rf(tune=True)
    candidates = list(ParameterGrid(grid3))
    print(f"  [pooled-OOF grid search MODEL 3 random_forest over {len(candidates)} configs ...]")
    grid_rows = []
    for params in candidates:
        ov = oof(clone(base3).set_params(**params), Xb_full)
        grid_rows.append({"params": params, "oof": ov,
                          "AUPR": average_precision_score(yb, ov),
                          "TPR5": tpr_at_fpr(yb, ov, 0.05),
                          "TPR1": tpr_at_fpr(yb, ov, 0.01)})
    rf_selected = {  # selection on POOLED out-of-fold values
        "MODEL3 rf [best AUPR]":   max(grid_rows, key=lambda r: r["AUPR"]),
        "MODEL3 rf [best TPR@5%]": max(grid_rows, key=lambda r: r["TPR5"]),
        "MODEL3 rf [best TPR@1%]": max(grid_rows, key=lambda r: r["TPR1"]),
    }
    for name, row in rf_selected.items():
        _rec(name, row["oof"], yb, val_results, val_scores)
        tuned_params[name] = _jsonable(row["params"])

    # MODEL 4 : XGBoost -- pooled-OOF grid search, selected 3 ways (if xgboost installed) 
    m4 = make_model4_xgb(cols, scale_pos_weight=spw)
    xgb_selected = {}
    if m4 is not None:
        cands4 = list(ParameterGrid(xgb_grid()))
        print(f"  [pooled-OOF grid search MODEL 4 xgboost over {len(cands4)} configs ...]")
        rows4 = []
        for params in cands4:
            ov = oof(clone(m4).set_params(**params), Xb_full)
            rows4.append({"params": params, "oof": ov,
                          "AUPR": average_precision_score(yb, ov),
                          "TPR5": tpr_at_fpr(yb, ov, 0.05),
                          "TPR1": tpr_at_fpr(yb, ov, 0.01)})
        xgb_selected = {
            "MODEL4 xgb [best AUPR]":   max(rows4, key=lambda r: r["AUPR"]),
            "MODEL4 xgb [best TPR@5%]": max(rows4, key=lambda r: r["TPR5"]),
            "MODEL4 xgb [best TPR@1%]": max(rows4, key=lambda r: r["TPR1"]),
        }
        for name, row in xgb_selected.items():
            _rec(name, row["oof"], yb, val_results, val_scores)
            tuned_params[name] = _jsonable(row["params"])

    _save_table(val_results, outdir, mode="val")
    val_pr = {k: val_scores[k] for k in (
        "af_pred (max rank-conf)", "narrowe_logreg_homology_all_featr",
        "MODEL3 rf [best AUPR]", "MODEL3 rf [best TPR@5%]", "MODEL3 rf [best TPR@1%]",
        "MODEL4 xgb [best AUPR]", "MODEL4 xgb [best TPR@5%]", "MODEL4 xgb [best TPR@1%]",
        ) if k in val_scores}
    _save_pr_curves(yb, val_pr, outdir, filename="validation_pr_curves.png",
                    title=f"Homodimer identification (validation: {cv_folds}-fold pooled OOF)")
    _export_params(tuned_params, val_results, outdir, cv_folds, seeds, cols)

    # ===================================================================== #
    # OPTIONAL - NESTED CV on "before" set: selection-bias-free estimate of the procedure
    #   "search the grid, pick best-AUPR config, train it". Inner folds select,
    #   outer folds grade; reported on pooled outer OOF with bootstrap CI.
    # ===================================================================== #
    if with_nested:
        print(f"\n################  NESTED CV ({outer_k} outer x {inner_k} inner, on 'before')"
              f"  ################")
        nested_results = {}

        print("  [MODEL 3 random_forest: nested CV (inner selects by AUPR) ...]")
        base3, grid3 = make_model3_rf(tune=True)
        _rec("MODEL3 rf [nested]",
             nested_cv_oof(base3, grid3, Xb_full, yb, outer_k, inner_k), yb, nested_results)
        if m4 is not None:
            print("  [MODEL 4 xgboost: nested CV (inner selects by AUPR) ...]")
            _rec("MODEL4 xgb [nested]",
                 nested_cv_oof(m4, xgb_grid(), Xb_full, yb, outer_k, inner_k), yb, nested_results)
        _save_table(nested_results, outdir, mode="nested")

    # ===================================================================== #
    # PHASE 2 - TEST: refit on the FULL before-set, score the after-set ONCE,
    #   with bootstrap CIs. The three selected RF configs are all evaluated.
    # ===================================================================== #
    print("\n################  TEST (trained on full 'before', scored on 'after')  ################")

    hdr("af_pred (max ranking confidence)")
    af = pd.to_numeric(after.get("max_rc"), errors="coerce")
    if af.isna().all():
        af = 0.8 * pd.to_numeric(after.get("max_iptm"), errors="coerce") \
             + 0.2 * pd.to_numeric(after.get("max_ptm"), errors="coerce")
    _rec("af_pred (max rank-conf)", af.values, ya, test_results, test_scores)

    for tag, with_h in [("narrowe_logreg_homology", True), ("narrowe_logreg_nohomology", False)]:
        hdr(tag)
        feats, pipe = make_narrowe_logreg(with_h)
        feats = [f for f in feats if f in cols]
        pipe.fit(Xb_full[feats], yb)
        _rec(tag, _score(pipe, Xa_full[feats]), ya, test_results, test_scores)

    for tag, feats in allfeat.items():
        hdr(f"{tag}  ({len(feats)} features)")
        pipe = _logreg_pipe().fit(Xb_full[feats], yb)
        _rec(tag, _score(pipe, Xa_full[feats]), ya, test_results, test_scores)

    # MODEL 3 : test ALL three selected RF configs (refit full before, seed-averaged)
    rf_cache = {}
    for name, row in rf_selected.items():
        hdr(f"{name}  ({len(seeds)}-seed average)   params={row['params']}")
        key = tuple(sorted(row["params"].items()))
        if key not in rf_cache:
            est = clone(base3).set_params(**row["params"])
            rf_cache[key] = seed_avg_scores(est, Xb_full, yb, Xa_full, seeds)
        _rec(name, rf_cache[key], ya, test_results, test_scores)

    # MODEL 4 : test ALL three selected XGB configs (refit full before, seed-averaged)
    if m4 is not None:
        xgb_cache = {}
        for name, row in xgb_selected.items():
            hdr(f"{name}  ({len(seeds)}-seed average)   params={row['params']}")
            key = tuple(sorted(row["params"].items()))
            if key not in xgb_cache:
                est = clone(m4).set_params(**row["params"])
                xgb_cache[key] = seed_avg_scores(est, Xb_full, yb, Xa_full, seeds)
            _rec(name, xgb_cache[key], ya, test_results, test_scores)
    else:
        print("\n[--] MODEL 4 xgboost SKIPPED  (install with: pip install xgboost)")

    _save_table(test_results, outdir, mode="test")
    pr = {k: test_scores[k] for k in (
        "af_pred (max rank-conf)", "narrowe_logreg_homology_all_featr",
        "MODEL3 rf [best AUPR]", "MODEL3 rf [best TPR@5%]", "MODEL3 rf [best TPR@1%]",
        "MODEL4 xgb [best AUPR]", "MODEL4 xgb [best TPR@5%]", "MODEL4 xgb [best TPR@1%]",
        ) if k in test_scores}
    _save_pr_curves(ya, pr, outdir)
    return test_results


def _save_table(results: dict, outdir: Path, mode="val"):
    rows = [{
        "method": name,
        "AUPR": round(m["AUPR"], 3),
        "pAUC@5%": round(m["pAUC@5%"], 3),
        "MCC": round(m["MCC"], 3),
        "TPR@5%FPR": round(m["TPR@5%FPR"], 3),
        "TPR@5%FPR_CI": f"[{m['TPR@5%FPR_CI'][0]:.3f}, {m['TPR@5%FPR_CI'][1]:.3f}]",
        "TPR@1%FPR": round(m["TPR@1%FPR"], 3),
        "TPR@1%FPR_CI": f"[{m['TPR@1%FPR_CI'][0]:.3f}, {m['TPR@1%FPR_CI'][1]:.3f}]",
    } for name, m in results.items()]
    tbl = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    spec = {
        "test":         ("test_results.csv",              "TEST"),
        "val":          ("validation_results.csv",        "VALIDATION"),
        "nested":       ("nested_cv_results.csv",         "NESTED CROSS-VALIDATION"),
        "nested_after": ("nested_after_only_results.csv", "NESTED CV (after-only)"),
    }
    if mode not in spec:
        raise ValueError(f"_save_table: unknown mode {mode!r}")
    fname, banner = spec[mode]
    print(f"\n---------------- {banner} ----------------")
    print(tbl.to_string(index=False))
    tbl.to_csv(outdir / fname, index=False)
    print(f"saved -> {outdir / fname}")


def _export_params(tuned_params: dict, val_results: dict, outdir: Path,
                   cv_folds: int, seeds, cols):
    """Export the chosen hyper-parameters + pooled-OOF validation metrics so the exact
    model configuration can be rebuilt and applied to new data."""
    payload = {
        "cv_folds": cv_folds,
        "seeds": list(seeds),
        "n_features": len(cols),
        "features": list(cols),
        "tuned_params": tuned_params,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tuned_params.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"exported parameters -> {outdir / 'tuned_params.json'}")


def _save_pr_curves(y_true, score_dict, outdir: Path,
                    filename="pr_curves.png",
                    title="Homodimer identification (test set)"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
    except Exception as e:  # pragma: no cover
        warnings.warn(f"matplotlib unavailable, skipping PR plot ({e})")
        return
    plt.figure(figsize=(6, 5))
    for name, s in score_dict.items():
        if s is None:
            continue
        s = np.asarray(s, dtype="float64")
        ok = ~np.isnan(s)
        p, r, _ = precision_recall_curve(np.asarray(y_true)[ok], s[ok])
        ap = average_precision_score(np.asarray(y_true)[ok], s[ok])
        plt.plot(r, p, label=f"{name} (AUPR={ap:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left", fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=150)
    print(f"saved -> {outdir / filename}")


# ============================================================================ #
# 6. MAIN
# ============================================================================ #
def nested_cv_after_only(after: pd.DataFrame, outdir: Path,
                         n_boot=2000, outer_k=5, inner_k=5):
    """Nested-CV evaluation on the AFTER-set only (honest AF features in train and test).
    All-feature LR, Random Forest, and XGBoost are each nested-CV'd (inner selects the
    config by AUPR; outer folds grade). af_pred and the Narrowe pruned LR are added as
    reference rows (plain pooled-OOF CV / fixed feature). Every row gets a bootstrap CI.
    """
    X = build_design(after)
    y = after["label"].astype(int).values
    cols = list(X.columns)
    spw = float((y == 0).sum()) / max(1, int((y == 1).sum()))
    skf = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=RNG)

    print(f"\n################  NESTED CV (after-only, {outer_k} outer x {inner_k} inner)  "
          f"################")
    print(f"after-set: {len(y)} proteins (pos={int(y.sum())}, neg={int((y==0).sum())})")

    results, scores = {}, {}

    def rec(name, s):
        s = np.asarray(s, dtype="float64")
        scores[name] = s
        m = all_metrics(y, s)
        lo1, hi1 = stratified_bootstrap_ci(y, s, lambda yt, ys: tpr_at_fpr(yt, ys, 0.01), n_boot=n_boot)
        lo5, hi5 = stratified_bootstrap_ci(y, s, lambda yt, ys: tpr_at_fpr(yt, ys, 0.05), n_boot=n_boot)
        m["TPR@1%FPR_CI"], m["TPR@5%FPR_CI"] = (lo1, hi1), (lo5, hi5)
        results[name] = m
        print(f"  {name:30s}  AUPR={m['AUPR']:.3f}  pAUC@5%={m['pAUC@5%']:.3f}  "
              f"MCC={m['MCC']:.3f}  TPR@5%={m['TPR@5%FPR']:.3f} "
              f"[{lo5:.3f},{hi5:.3f}]  TPR@1%={m['TPR@1%FPR']:.3f} [{lo1:.3f},{hi1:.3f}]")
    def plain_oof(est, Xsub):
        return cross_val_predict(clone(est), Xsub, y, cv=skf,
                                 method="predict_proba", n_jobs=4)[:, 1]

    # --- reference baselines ---
    af = pd.to_numeric(after.get("max_rc"), errors="coerce")
    if af.isna().all():
        af = 0.8 * pd.to_numeric(after.get("max_iptm"), errors="coerce") \
             + 0.2 * pd.to_numeric(after.get("max_ptm"), errors="coerce")
    rec("af_pred (max rank-conf)", af.values)                     # fixed feature, no CV

    for tag, with_h in [("narrowe_logreg_homology", True), ("narrowe_logreg_nohomology", False)]:
        feats, pipe = make_narrowe_logreg(with_h)
        feats = [f for f in feats if f in cols]
        rec(tag, plain_oof(pipe, X[feats]))          # fixed config -> plain CV

    # --- nested-CV models (inner selection by AUPR) ---
    print("  [logreg_all: nested CV ...]")
    base_lr, grid_lr = make_logreg_all(tune=True)
    hom_cols = homology_design_columns(cols)
    allfeat = {
        "logreg_all [nested]":            cols,
        "logreg_all_nohomology [nested]": [c for c in cols if c not in hom_cols],
    }
    for tag, feats in allfeat.items():
        rec(tag, nested_cv_oof(base_lr, grid_lr, X[feats], y, outer_k, inner_k))   # tuned -> nested

    print("  [random_forest: nested CV ...]")
    base_rf, grid_rf = make_model3_rf(tune=True)
    rec("random_forest [nested]", nested_cv_oof(base_rf, grid_rf, X, y, outer_k, inner_k))

    m4 = make_model4_xgb(cols, scale_pos_weight=spw)
    if m4 is not None:
        print("  [xgboost: nested CV ...]")
        rec("xgboost [nested]", nested_cv_oof(m4, xgb_grid(), X, y, outer_k, inner_k))
    else:
        print("  [--] xgboost SKIPPED  (install with: pip install xgboost)")

    _save_table(results, outdir, mode="nested_after")
    _save_pr_curves(y, scores, outdir, filename="nested_after_only_pr.png",
                    title=f"Homodimer identification (after-only nested CV, {outer_k}x{inner_k})")
    return results


def dataset_stats(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Print and return the composition of a split AFTER all preprocessing steps:
    surviving rows, positive/negative balance, the homodimer / heterodimer / monomer
    breakdown (from the labels file's 'stoichiometry' column when present), and how many
    rows carry the no_homolog flag."""
    n = len(df)
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    rows = [("rows remaining", n),
            ("positives (label=1)", pos),
            ("negatives (label=0)", neg)]

    if "stoichiometry" in df.columns:
        stoich = df["stoichiometry"].astype(str).str.strip().str.lower()
        for cls in ("homodimer", "heterodimer", "monomer"):
            rows.append((f"  {cls}", int((stoich == cls).sum()))) 
        other = int((~stoich.isin(["homodimer", "heterodimer", "monomer"])).sum())
        if other:
            rows.append(("  other/unlabelled", other))
    else:
        rows.append(("  stoichiometry", "not in labels file"))

    if "no_homolog" in df.columns:
        rows.append(("no_homolog flag = 1", int((df["no_homolog"] == 1).sum())))

    print(f"\n---------------- DATA STATISTICS after preprocessing [{split_name}] "
          f"----------------")
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k:<{width}} : {v}")
    return pd.DataFrame(rows, columns=["statistic", "value"]).assign(split=split_name)


def save_preprocessed(df: pd.DataFrame, outdir: Path, split_name: str) -> Path:
    """Write the preprocessed split to disk so the exact modelling input is reproducible."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"preprocessed_{split_name}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"saved preprocessed {split_name} ({len(df)} rows x {df.shape[1]} cols) -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-before", type=Path, default=None)
    ap.add_argument("--features-after", type=Path, required=True)
    ap.add_argument("--labels-before", type=Path, default=None)
    ap.add_argument("--labels-after", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--after-only-nested", action="store_true",
                    help="run nested CV on the AFTER-set only (LR/RF/XGB); skips before/after")
    ap.add_argument("--cv-folds", type=int, default=5,
                    help="stratified k for the pooled out-of-fold validation on 'before'")
    ap.add_argument("--nested-cv-val", action="store_true",
                    help="also run nested CV on the development set (selection-unbiased estimate of RF/XGB)")
    ap.add_argument("--outer-k", type=int, default=5, help="nested-CV outer folds (grading)")
    ap.add_argument("--inner-k", type=int, default=5, help="nested-CV inner folds (selection)")
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="number of seeds to average stochastic models over")
    args = ap.parse_args()

    adf = _read_features(args.features_after)
    labels_after = pd.read_csv(args.labels_after)
    if "label" not in labels_after.columns or "id" not in labels_after.columns:
        ap.error("--labels-after must have columns 'id' and 'label'")
    after = preprocess(load_df(adf, labels_after, "after"), "after")
    stats = [dataset_stats(after, "after")]
    save_preprocessed(after, args.outdir, "after")

    if args.after_only_nested:
        pd.concat(stats).to_csv(args.outdir / "dataset_statistics.csv", index=False)
        print(f"saved -> {args.outdir / 'dataset_statistics.csv'}")
        print(f"\nloaded: after={len(after)} rows (pos={after.label.sum()})  [after-only nested CV]")
        nested_cv_after_only(after, args.outdir, n_boot=args.n_boot,
                             outer_k=args.outer_k, inner_k=args.inner_k)
        print("\nDone.")
        return

    if args.features_before is None or args.labels_before is None:
        ap.error("--features-before and --labels-before are required "
                 "unless --after-only-nested is set")
    bdf = _read_features(args.features_before)
    labels_before = pd.read_csv(args.labels_before)
    if "label" not in labels_before.columns or "id" not in labels_before.columns:
        ap.error("--labels-before must have columns 'id' and 'label'")
    before = preprocess(load_df(bdf, labels_before, "before"), "before")
    stats.append(dataset_stats(before, "before"))
    save_preprocessed(before, args.outdir, "before")
    pd.concat(stats).to_csv(args.outdir / "dataset_statistics.csv", index=False)
    print(f"saved -> {args.outdir / 'dataset_statistics.csv'}")

    print(f"\nloaded: before={len(before)} rows (pos={before.label.sum()}), "
          f"after={len(after)} rows (pos={after.label.sum()})")

    evaluate_all(before, after, args.outdir, n_boot=args.n_boot,
                 n_seeds=args.n_seeds, cv_folds=args.cv_folds,
                 with_nested=args.nested_cv_val, outer_k=args.outer_k, inner_k=args.inner_k)
    print("\nDone.")


if __name__ == "__main__":
    main()
