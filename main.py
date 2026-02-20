#!/usr/bin/env python3
"""Survival model runner with Optuna optimization (TPE/NSGA-II)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set
import warnings

import joblib
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.pruners import MedianPruner
from joblib import Parallel, delayed
from loguru import logger
from sklearn.model_selection import train_test_split

from survbase.data import build_survival_y, get_dataframe
from survbase.metrics import concordance_index, integrated_brier_score, one_calibration
from survbase.statistical_based_models import AFT, CoxPH, LogisticHazard, PCHazard
from survbase.cluster_based_models import DeepCoxMixtures, SCA, VaDeSC, DVCSurv
from survbase.preprocessing import TabularPreprocessor, TabularPreprocessorConfig
from survbase.ml_based_models import SurvivalForest, XGBoostCox,  DeepHit, DeepSurv
from survbase.converse_siamese import CONVERSE_siamese
from survbase.converse_single import CONVERSE_single

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

try:
    from survbase.data import DATA_DIR as _DATA_DIR
except Exception:
    _DATA_DIR = None


@dataclass
class ExperimentConfig:
    datasets: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    n_bootstrap_iters: int = 10
    seed: int = 42
    val_size: float = 0.2
    test_size: float = 0.2
    event_col: str = "event"
    time_col: str = "time"
    n_trials: int = 50
    n_startup_trials: int = 10
    n_jobs: int = -1
    space_dir: Optional[str] = None
    output: str = "results_{time}.csv"
    interpolation: str = "step"
    log_level: str = "INFO"
    optimization_metrics: List[str] = field(default_factory=lambda: ["cindex", "ibs", "cal"])
    optimizer: str = "tpe"
    pruning: bool = True
    storage: Optional[str] = None
    save_final_model: bool = False



def _cpu_count() -> int:
    try:
        import multiprocessing as mp

        return mp.cpu_count()
    except Exception:
        return 1


def _effective_n_jobs(n_jobs: int) -> int:
    if n_jobs is None or int(n_jobs) == 0:
        return 1
    if int(n_jobs) < 0:
        return max(1, _cpu_count())
    return int(n_jobs)


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _discover_all_datasets() -> List[str]:
    if _DATA_DIR and os.path.isdir(_DATA_DIR):
        names = [
            os.path.splitext(fn)[0] for fn in os.listdir(_DATA_DIR) if fn.lower().endswith(".csv")
        ]
        names.sort()
        return names
    logger.error("Cannot discover datasets for --dataset all: survbase.data.DATA_DIR not found.")
    return []


def _normalize_datasets(arg_val: Optional[Sequence[str]]) -> List[str]:
    if not arg_val:
        return []
    chunks: List[str] = []
    for item in arg_val:
        chunks.extend([c.strip() for c in str(item).split(",") if c.strip()])
    if len(chunks) == 1 and chunks[0].lower() == "all":
        return _discover_all_datasets()
    seen: Set[str] = set()
    out: List[str] = []
    for c in chunks:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser(description="Survival model optimization with Optuna.")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--dataset", type=str, nargs="+")
    p.add_argument("--models", type=str, nargs="+")
    p.add_argument("--n-bootstrap-iters", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--val-size", type=float)
    p.add_argument("--test-size", type=float)
    p.add_argument("--event-col", type=str)
    p.add_argument("--time-col", type=str)
    p.add_argument("--n-trials", type=int)
    p.add_argument("--n-startup-trials", type=int)
    p.add_argument("--n-jobs", type=int)
    p.add_argument("--space-dir", type=str)
    p.add_argument("--output", type=str)
    p.add_argument("--interpolation", type=str, choices=["step", "linear"])
    p.add_argument(
        "--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "SUCCESS"]
    )
    p.add_argument("--optimizer", type=str, choices=["tpe", "nsga"])
    p.add_argument("--no-pruning", action="store_true")
    p.add_argument("--storage", type=str)
    p.add_argument("--save-final-model", action=argparse.BooleanOptionalAction, default=None)
    args = p.parse_args()

    cfg = ExperimentConfig()

    if args.config and os.path.exists(args.config):
        cfg_yaml = _load_yaml(args.config)
        for k, v in (cfg_yaml or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    if args.dataset is not None:
        cfg.datasets = _normalize_datasets(args.dataset)
    if args.models is not None:
        cfg.models = list(args.models)

    for name in [
        "n_bootstrap_iters",
        "seed",
        "val_size",
        "test_size",
        "event_col",
        "time_col",
        "n_trials",
        "n_startup_trials",
        "n_jobs",
        "space_dir",
        "output",
        "interpolation",
        "log_level",
        "optimizer",
        "storage",
        "save_final_model",    
    ]:
        v = getattr(args, name, None)
        if v is not None:
            setattr(cfg, name, v)

    if args.no_pruning:
        cfg.pruning = False

    if not cfg.datasets:
        p.error("Please provide --dataset (one or many) or set in --config.")
    if not cfg.models:
        p.error("Please provide --models (or set in --config).")

    cfg.log_level = (cfg.log_level or "INFO").upper()

    _now = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    cfg.output = (cfg.output or "results_{time}.csv").format(dataset="{dataset}", time=_now)
    if not os.path.isabs(cfg.output):
        cfg.output = os.path.join("results", cfg.output)

    n_eff = _effective_n_jobs(cfg.n_jobs)
    logger.info(f"n_jobs={cfg.n_jobs} -> effective={n_eff} (CPUs={_cpu_count()})")
    logger.info(f"Optimization metrics: {cfg.optimization_metrics}")

    return cfg


MODEL_REGISTRY = {
    "coxph": CoxPH,
    "aft": AFT,
    "pchazard": PCHazard,
    "deepsurv": DeepSurv,
    "logistic_hazard": LogisticHazard,
    "survival_forest": SurvivalForest,
    "xgbcox": XGBoostCox,
    "dvcsurv": DVCSurv,
    "deephit": DeepHit,
    "sca": SCA,
    "vadesc": VaDeSC,
    "dcm": DeepCoxMixtures,
    "converse_siamese": CONVERSE_siamese,
    "converse_single": CONVERSE_single,
}


def make_estimator(name: str, params: Dict[str, Any], default_interp: str) -> Any:
    cls = MODEL_REGISTRY[name]
    p = dict(params or {})
    p.setdefault("interpolation", default_interp)
    return cls(**p)


def set_global_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def suggest_params_from_space(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    for name, spec in (space or {}).items():
        if isinstance(spec, dict):
            if "choice" in spec:
                params[name] = trial.suggest_categorical(name, spec["choice"])
            elif "uniform" in spec:
                lo, hi = spec["uniform"]
                params[name] = trial.suggest_float(name, float(lo), float(hi))
            elif "loguniform" in spec:
                lo, hi = spec["loguniform"]
                params[name] = trial.suggest_float(name, float(lo), float(hi), log=True)
            elif "int" in spec:
                lo, hi = spec["int"]
                params[name] = trial.suggest_int(name, int(lo), int(hi))
            elif "bool" in spec:
                params[name] = trial.suggest_categorical(name, [True, False])
            elif "fixed" in spec:
                params[name] = spec["fixed"]
            else:
                logger.warning(f"Unknown spec for {name}: {spec}")
        elif isinstance(spec, list):
            params[name] = trial.suggest_categorical(name, spec)
        else:
            params[name] = spec

    if "clustering_method" in params: 
        if params["clustering_method"] == "agglomerative" and "linkage" not in params:
            params["linkage"] = trial.suggest_categorical("linkage", ["ward", "complete", "average", "single"])
        elif params["clustering_method"] == "GaussianMixture" and "covariance_type" not in params:
            params["covariance_type"] = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])
        elif params["clustering_method"] == "SpectralClustering" and "affinity" not in params:
            params["affinity"] = trial.suggest_categorical("affinity", ["nearest_neighbors", "rbf"])
    return params


def _fit_score_one_bootstrap(
    model_name: str,
    params: Dict[str, Any],
    df_full: pd.DataFrame,
    val_size: float,
    test_size: float,
    event_col: str,
    time_col: str,
    default_interp: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Perform one bootstrap iteration:
    - Split data into train/val/test
    - Preprocess each split
    - Fit model on train
    - Compute metrics on train/val/test
    """
    set_global_seed(seed)

    params = dict(params)
    params.setdefault("random_state", int(seed))

    # Three-way split: first split off test, then split remaining into train/val
    n_total = len(df_full)

    # Try stratified split based on event status
    try:
        stratify = df_full[event_col].astype(bool).to_numpy()

        # First split: train+val vs test
        trainval_idx, test_idx = train_test_split(
            np.arange(n_total),
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )

        # Second split: train vs val
        # Adjust val_size to be relative to the train+val set
        val_size_adjusted = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=val_size_adjusted,
            random_state=seed + 1,
            stratify=stratify[trainval_idx],
        )
    except ValueError:
        # Fall back to non-stratified if stratification fails
        trainval_idx, test_idx = train_test_split(
            np.arange(n_total),
            test_size=test_size,
            random_state=seed,
        )
        val_size_adjusted = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=val_size_adjusted,
            random_state=seed + 1,
        )

    df_tr = df_full.iloc[train_idx].reset_index(drop=True)
    df_va = df_full.iloc[val_idx].reset_index(drop=True)
    df_te = df_full.iloc[test_idx].reset_index(drop=True)

    # Preprocess data
    prep = TabularPreprocessor(TabularPreprocessorConfig(event_col=event_col, time_col=time_col))
    X_tr = prep.fit_transform(df_tr)
    X_va = prep.transform(df_va)
    X_te = prep.transform(df_te)

    y_tr = build_survival_y(df_tr, event_col=event_col, time_col=time_col)
    y_va = build_survival_y(df_va, event_col=event_col, time_col=time_col)
    y_te = build_survival_y(df_te, event_col=event_col, time_col=time_col)

    # Fit model
    est = make_estimator(model_name, params, default_interp)

    t0 = time.perf_counter()

    # Pre-training for dvcsurv based models
    if "converse" in model_name or "dvcsurv" in model_name:
        # Stage A: warmup pre-training
        # turn off clustering + all contrastive losses
        est.pretraining = True
        est.warm_start = False
        est.fit(X_tr, y_tr,)
        # Stage B: run K-means on embeddings to set fixed centers
        est.initialize_centers(X_tr, batch_size=est.batch_size, random_state=int(seed))
        # Stage C: full training with clustering + contrastive
        est.pretraining = False
        est.warm_start = True

    est.fit(X_tr, y_tr)
    dur = time.perf_counter() - t0

    # Compute metrics on all three splits
    results = {"time_sec": dur}

    for split_name, X_split, y_split in [
        ("train", X_tr, y_tr),
        ("val", X_va, y_va),
        ("test", X_te, y_te),
    ]:
        try:
            preds = est.predict(X_split)
            risks = est.predict_risk(X_split)
            cidx = float(concordance_index(y_split, risks))
            ibs = float(integrated_brier_score(y_tr, y_split, preds))
            cal = float(one_calibration(y_split, preds))
        except Exception as e:
            logger.warning(f"Metric computation failed for {split_name}: {e}")
            cidx, ibs, cal = float("nan"), float("nan"), float("nan")

        results[f"{split_name}_cindex"] = cidx
        results[f"{split_name}_ibs"] = ibs
        results[f"{split_name}_cal"] = cal

    return results


def _compute_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval using percentile method."""
    if len(values) == 0 or np.all(np.isnan(values)):
        return (float("nan"), float("nan"))

    alpha = 1 - confidence
    lower_p = (alpha / 2) * 100
    upper_p = (1 - alpha / 2) * 100

    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return (float("nan"), float("nan"))

    lower = float(np.percentile(valid_values, lower_p))
    upper = float(np.percentile(valid_values, upper_p))
    return (lower, upper)


def evaluate_candidate_bootstrap(
    params: Dict[str, Any],
    model_name: str,
    df_full: pd.DataFrame,
    event_col: str,
    time_col: str,
    n_bootstrap_iters: int,
    val_size: float,
    test_size: float,
    seed: int,
    default_interp: str,
    n_jobs: int,
    trial: Optional[optuna.Trial] = None,
    optimization_metrics: List[str] = None,
    csv_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate candidate parameters using bootstrap iterations.

    For each iteration:
    - Split data into train/val/test with different seeds
    - Fit model and compute metrics
    - Log metrics to CSV

    Returns aggregated metrics for Optuna optimization.
    """
    if optimization_metrics is None:
        optimization_metrics = ["cindex", "ibs", "cal"]

    # Run bootstrap iterations with different seeds
    bootstrap_results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_fit_score_one_bootstrap)(
            model_name,
            params,
            df_full,
            val_size,
            test_size,
            event_col,
            time_col,
            default_interp,
            seed=seed + i * 10007,  # Different seed for each iteration
        )
        for i in range(n_bootstrap_iters)
    )

    # Extract metrics for each split
    train_cindexs = np.array([r["train_cindex"] for r in bootstrap_results], dtype=float)
    train_ibss = np.array([r["train_ibs"] for r in bootstrap_results], dtype=float)
    train_cals = np.array([r["train_cal"] for r in bootstrap_results], dtype=float)

    val_cindexs = np.array([r["val_cindex"] for r in bootstrap_results], dtype=float)
    val_ibss = np.array([r["val_ibs"] for r in bootstrap_results], dtype=float)
    val_cals = np.array([r["val_cal"] for r in bootstrap_results], dtype=float)

    test_cindexs = np.array([r["test_cindex"] for r in bootstrap_results], dtype=float)
    test_ibss = np.array([r["test_ibs"] for r in bootstrap_results], dtype=float)
    test_cals = np.array([r["test_cal"] for r in bootstrap_results], dtype=float)

    times = np.array([r["time_sec"] for r in bootstrap_results], dtype=float)

    # Log each iteration to CSV if path provided
    if csv_output_path is not None:
        for i, result in enumerate(bootstrap_results):
            row = {
                "bootstrap_iter": i,
                "params": json.dumps(params),
                "train_cindex": result["train_cindex"],
                "train_ibs": result["train_ibs"],
                "train_cal": result["train_cal"],
                "val_cindex": result["val_cindex"],
                "val_ibs": result["val_ibs"],
                "val_cal": result["val_cal"],
                "test_cindex": result["test_cindex"],
                "test_ibs": result["test_ibs"],
                "test_cal": result["test_cal"],
                "time_sec": result["time_sec"],
            }
            pd.DataFrame([row]).to_csv(csv_output_path, index=False, mode="a", header=False)

    # Optuna pruning based on validation C-index (if single-objective)
    if trial is not None and len(optimization_metrics) == 1:
        intermediate_score = float(np.nanmean(val_cindexs))
        trial.report(intermediate_score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Compute confidence intervals
    train_cindex_ci = _compute_ci(train_cindexs)
    train_ibs_ci = _compute_ci(train_ibss)
    train_cal_ci = _compute_ci(train_cals)

    val_cindex_ci = _compute_ci(val_cindexs)
    val_ibs_ci = _compute_ci(val_ibss)
    val_cal_ci = _compute_ci(val_cals)

    test_cindex_ci = _compute_ci(test_cindexs)
    test_ibs_ci = _compute_ci(test_ibss)
    test_cal_ci = _compute_ci(test_cals)

    return {
        # Train metrics
        "train_cindex_mean": float(np.nanmean(train_cindexs)),
        "train_cindex_std": float(np.nanstd(train_cindexs)),
        "train_cindex_ci": train_cindex_ci,
        "train_ibs_mean": float(np.nanmean(train_ibss)),
        "train_ibs_std": float(np.nanstd(train_ibss)),
        "train_ibs_ci": train_ibs_ci,
        "train_cal_mean": float(np.nanmean(train_cals)),
        "train_cal_std": float(np.nanstd(train_cals)),
        "train_cal_ci": train_cal_ci,
        # Validation metrics
        "val_cindex_mean": float(np.nanmean(val_cindexs)),
        "val_cindex_std": float(np.nanstd(val_cindexs)),
        "val_cindex_ci": val_cindex_ci,
        "val_ibs_mean": float(np.nanmean(val_ibss)),
        "val_ibs_std": float(np.nanstd(val_ibss)),
        "val_ibs_ci": val_ibs_ci,
        "val_cal_mean": float(np.nanmean(val_cals)),
        "val_cal_std": float(np.nanstd(val_cals)),
        "val_cal_ci": val_cal_ci,
        # Test metrics
        "test_cindex_mean": float(np.nanmean(test_cindexs)),
        "test_cindex_std": float(np.nanstd(test_cindexs)),
        "test_cindex_ci": test_cindex_ci,
        "test_ibs_mean": float(np.nanmean(test_ibss)),
        "test_ibs_std": float(np.nanstd(test_ibss)),
        "test_ibs_ci": test_ibs_ci,
        "test_cal_mean": float(np.nanmean(test_cals)),
        "test_cal_std": float(np.nanstd(test_cals)),
        "test_cal_ci": test_cal_ci,
        # Timing
        "time_sec_total": float(np.nansum(times)),
    }


class LoguruProgressCallback:
    """Optuna callback to log optimization progress with loguru."""

    def __init__(self, n_trials: int, log_interval_pct: int = 10):
        self.n_trials = n_trials
        self.log_interval = max(1, n_trials * log_interval_pct // 100)

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.number % self.log_interval == 0 or trial.number == self.n_trials - 1:
            progress = (trial.number + 1) / self.n_trials * 100

            if len(study.directions) > 1:
                if trial.values is None:
                    logger.info(
                        f"Optuna Progress: {progress:.1f}% ({trial.number + 1}/{self.n_trials}) | "
                        f"Current trial has no values."
                    )
                else:
                    logger.info(
                        f"Optuna Progress: {progress:.1f}% ({trial.number + 1}/{self.n_trials}) | "
                        f"Current values: {[f'{v:.4f}' for v in trial.values]}"
                    )
            elif study.direction == optuna.study.StudyDirection.MAXIMIZE:
                logger.info(
                    f"Optuna Progress: {progress:.1f}% ({trial.number + 1}/{self.n_trials}) | "
                    f"Best value: {study.best_value:.4f} | Current value: {trial.value:.4f}"
                )
            else:
                logger.info(
                    f"Optuna Progress: {progress:.1f}% ({trial.number + 1}/{self.n_trials}) | "
                    f"Best value: {study.best_value:.4f} | Current value: {trial.value:.4f}"
                )


def optuna_optimize(
    model_name: str,
    space: Dict[str, Any],
    df_full: pd.DataFrame,
    event_col: str,
    time_col: str,
    n_bootstrap_iters: int,
    val_size: float,
    test_size: float,
    n_trials: int,
    n_startup_trials: int,
    seed: int,
    default_interp: str,
    n_jobs: int,
    optimization_metrics: List[str],
    optimizer: str,
    enable_pruning: bool,
    storage: Optional[str],
    csv_output_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Optimize hyperparameters using Optuna with bootstrap evaluation.

    For each trial:
    - Sample hyperparameters
    - Evaluate using bootstrap iterations (multiple train/val/test splits)
    - Return average validation metrics to Optuna
    """

    def objective(trial: optuna.Trial):
        trial_seed = seed + trial.number * 10007
        set_global_seed(trial_seed)

        params = suggest_params_from_space(trial, space)

        try:
            results = evaluate_candidate_bootstrap(
                params=params,
                model_name=model_name,
                df_full=df_full,
                event_col=event_col,
                time_col=time_col,
                n_bootstrap_iters=n_bootstrap_iters,
                val_size=val_size,
                test_size=test_size,
                seed=trial_seed,
                default_interp=default_interp,
                n_jobs=n_jobs,
                trial=trial if enable_pruning else None,
                optimization_metrics=optimization_metrics,
                csv_output_path=csv_output_path,
            )

            # Extract validation metrics for optimization
            val_cindex = results["val_cindex_mean"]
            val_ibs = results["val_ibs_mean"]
            val_cal = results["val_cal_mean"]

            if np.isnan(val_cindex) or np.isnan(val_ibs) or np.isnan(val_cal):
                raise optuna.TrialPruned()

            # Return metrics based on optimization_metrics configuration
            metrics_to_return = []
            for metric in optimization_metrics:
                if metric == "cindex":
                    metrics_to_return.append(val_cindex)
                elif metric == "ibs":
                    metrics_to_return.append(val_ibs)
                elif metric == "cal":
                    metrics_to_return.append(val_cal)

            # If only one metric, return as scalar; otherwise return as tuple
            if len(metrics_to_return) == 1:
                return metrics_to_return[0]
            else:
                return tuple(metrics_to_return)

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            raise optuna.TrialPruned()

    # Determine optimization directions based on metrics
    directions = []
    for metric in optimization_metrics:
        if metric == "cindex":
            directions.append("maximize")
        elif metric in ["ibs", "cal"]:
            directions.append("minimize")
        else:
            logger.warning(f"Unknown metric {metric}, defaulting to maximize")
            directions.append("maximize")

    # Choose sampler based on optimizer configuration
    optimizer = optimizer.lower()
    if optimizer == "tpe":
        # TPE supports both single and multi-objective optimization
        if len(optimization_metrics) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=optuna.exceptions.ExperimentalWarning)
                sampler = TPESampler(
                    seed=seed, n_startup_trials=n_startup_trials, multivariate=True
                )
            logger.info(
                f"Using TPE sampler for multi-objective optimization ({optimization_metrics})"
            )
        else:
            sampler = TPESampler(seed=seed, n_startup_trials=n_startup_trials)
            logger.info(
                f"Using TPE sampler for single-objective optimization ({optimization_metrics[0]})"
            )
    elif optimizer == "nsga":
        # NSGA-II for multi-objective optimization
        sampler = NSGAIISampler(seed=seed)
        if len(optimization_metrics) > 1:
            logger.info(
                f"Using NSGA-II sampler for {len(optimization_metrics)}-objective optimization ({optimization_metrics})"
            )
        else:
            logger.info(
                f"Using NSGA-II sampler for single-objective optimization ({optimization_metrics[0]})"
            )
    else:
        logger.warning(f"Unknown optimizer '{optimizer}', defaulting to TPE")
        sampler = TPESampler(seed=seed, n_startup_trials=n_startup_trials)
        logger.info(f"Using TPE sampler (default) for optimization ({optimization_metrics})")

    pruner = (
        MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=0)
        if enable_pruning
        else optuna.pruners.NopPruner()
    )

    study_name = f"{model_name}_{seed}"
    study = optuna.create_study(
        study_name=study_name,
        directions=directions,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    logger.info(
        f"Starting optimization with {n_trials} trials, {n_bootstrap_iters} bootstrap iterations per trial"
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False,
        callbacks=[LoguruProgressCallback(n_trials, log_interval_pct=10)],
    )

    # Extract best parameters
    if len(optimization_metrics) > 1:
        pareto_trials = study.best_trials
        logger.info(f"Found {len(pareto_trials)} Pareto-optimal solutions")

        # For multi-objective, select best based on first metric (usually C-index)
        best_trial = max(pareto_trials, key=lambda t: t.values[0])
        best_params = best_trial.params

        metric_str = ", ".join(
            [f"{m}={v:.4f}" for m, v in zip(optimization_metrics, best_trial.values)]
        )
        logger.info(f"Best trial (by {optimization_metrics[0]}): {metric_str}")
    else:
        best_params = study.best_params
        logger.info(f"Best {optimization_metrics[0]}: {study.best_value:.4f}")

    return best_params, study


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    cfg = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if threadpool_limits is not None:
        try:
            threadpool_limits(1)
        except Exception:
            pass

    n_eff = _effective_n_jobs(cfg.n_jobs)

    os.makedirs(os.path.dirname(cfg.output) or ".", exist_ok=True)
    logger.info(f"Output template: {cfg.output}")

    for dataset_name in cfg.datasets:
        # Summary CSV for aggregated results
        summary_path = cfg.output.format(dataset=dataset_name)
        if not os.path.exists(summary_path):
            pd.DataFrame(
                [],
                columns=[
                    "dataset",
                    "model",
                    "seed",
                    "best_params",
                    "train_cindex_mean",
                    "train_cindex_std",
                    "train_cindex_ci_lower",
                    "train_cindex_ci_upper",
                    "train_ibs_mean",
                    "train_ibs_std",
                    "train_ibs_ci_lower",
                    "train_ibs_ci_upper",
                    "train_cal_mean",
                    "train_cal_std",
                    "train_cal_ci_lower",
                    "train_cal_ci_upper",
                    "val_cindex_mean",
                    "val_cindex_std",
                    "val_cindex_ci_lower",
                    "val_cindex_ci_upper",
                    "val_ibs_mean",
                    "val_ibs_std",
                    "val_ibs_ci_lower",
                    "val_ibs_ci_upper",
                    "val_cal_mean",
                    "val_cal_std",
                    "val_cal_ci_lower",
                    "val_cal_ci_upper",
                    "test_cindex_mean",
                    "test_cindex_std",
                    "test_cindex_ci_lower",
                    "test_cindex_ci_upper",
                    "test_ibs_mean",
                    "test_ibs_std",
                    "test_ibs_ci_lower",
                    "test_ibs_ci_upper",
                    "test_cal_mean",
                    "test_cal_std",
                    "test_cal_ci_lower",
                    "test_cal_ci_upper",
                    "time_sec_total",
                    "n_bootstrap_iters",
                    "n_trials",
                    "space_file",
                    "status",
                    "error",
                ],
            ).to_csv(summary_path, index=False)
        logger.info(f"[{dataset_name}] summary CSV -> {summary_path}")

        # Detailed CSV for bootstrap iterations (one row per iteration)
        detailed_path = summary_path.replace(".csv", "_detailed.csv")
        if not os.path.exists(detailed_path):
            pd.DataFrame(
                [],
                columns=[
                    "bootstrap_iter",
                    "params",
                    "train_cindex",
                    "train_ibs",
                    "train_cal",
                    "val_cindex",
                    "val_ibs",
                    "val_cal",
                    "test_cindex",
                    "test_ibs",
                    "test_cal",
                    "time_sec",
                ],
            ).to_csv(detailed_path, index=False)
        logger.info(f"[{dataset_name}] detailed CSV -> {detailed_path}")

        df = get_dataframe(dataset_name)
        logger.info(f"[{dataset_name}] shape={df.shape}")

        for model_name in cfg.models:
            logger.info(f"[{dataset_name}] Model: {model_name}")

            space_path = ""
            space: Dict[str, Any] = {}
            if cfg.space_dir:
                formatted_space_dir = cfg.space_dir.format(dataset=dataset_name)
                for candidate in [
                    os.path.join(formatted_space_dir, f"{model_name}.yaml"),
                    os.path.join(formatted_space_dir, f"{model_name.lower()}.yaml"),
                ]:
                    if os.path.exists(candidate):
                        space_path = candidate
                        break
                if space_path:
                    loaded = _load_yaml(space_path)
                    space = loaded.get("params") if "params" in loaded else loaded
                    logger.info(f"[{dataset_name}] space <- {space_path}")
                else:
                    logger.info(f"[{dataset_name}] no space file; using defaults")

            t0 = time.perf_counter()
            try:
                best_params, study = optuna_optimize(
                    model_name=model_name,
                    space=space,
                    df_full=df,
                    event_col=cfg.event_col,
                    time_col=cfg.time_col,
                    n_bootstrap_iters=cfg.n_bootstrap_iters,
                    val_size=cfg.val_size,
                    test_size=cfg.test_size,
                    n_trials=cfg.n_trials,
                    n_startup_trials=cfg.n_startup_trials,
                    seed=cfg.seed,
                    default_interp=cfg.interpolation,
                    n_jobs=n_eff,
                    optimization_metrics=cfg.optimization_metrics,
                    optimizer=cfg.optimizer,
                    enable_pruning=cfg.pruning,
                    storage=cfg.storage,
                    csv_output_path=detailed_path,
                )
                optim_time = time.perf_counter() - t0

                # Re-evaluate best params to get all metrics
                logger.info(
                    f"[{dataset_name}] Re-evaluating best parameters on bootstrap iterations"
                )
                final_results = evaluate_candidate_bootstrap(
                    params=best_params,
                    model_name=model_name,
                    df_full=df,
                    event_col=cfg.event_col,
                    time_col=cfg.time_col,
                    n_bootstrap_iters=cfg.n_bootstrap_iters,
                    val_size=cfg.val_size,
                    test_size=cfg.test_size,
                    seed=cfg.seed,
                    default_interp=cfg.interpolation,
                    n_jobs=n_eff,
                    trial=None,
                    optimization_metrics=cfg.optimization_metrics,
                    csv_output_path=None,  # Don't log again
                )

                # Build summary row
                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": cfg.seed,
                    "best_params": json.dumps(best_params),
                    # Train metrics
                    "train_cindex_mean": final_results["train_cindex_mean"],
                    "train_cindex_std": final_results["train_cindex_std"],
                    "train_cindex_ci_lower": final_results["train_cindex_ci"][0],
                    "train_cindex_ci_upper": final_results["train_cindex_ci"][1],
                    "train_ibs_mean": final_results["train_ibs_mean"],
                    "train_ibs_std": final_results["train_ibs_std"],
                    "train_ibs_ci_lower": final_results["train_ibs_ci"][0],
                    "train_ibs_ci_upper": final_results["train_ibs_ci"][1],
                    "train_cal_mean": final_results["train_cal_mean"],
                    "train_cal_std": final_results["train_cal_std"],
                    "train_cal_ci_lower": final_results["train_cal_ci"][0],
                    "train_cal_ci_upper": final_results["train_cal_ci"][1],
                    # Val metrics
                    "val_cindex_mean": final_results["val_cindex_mean"],
                    "val_cindex_std": final_results["val_cindex_std"],
                    "val_cindex_ci_lower": final_results["val_cindex_ci"][0],
                    "val_cindex_ci_upper": final_results["val_cindex_ci"][1],
                    "val_ibs_mean": final_results["val_ibs_mean"],
                    "val_ibs_std": final_results["val_ibs_std"],
                    "val_ibs_ci_lower": final_results["val_ibs_ci"][0],
                    "val_ibs_ci_upper": final_results["val_ibs_ci"][1],
                    "val_cal_mean": final_results["val_cal_mean"],
                    "val_cal_std": final_results["val_cal_std"],
                    "val_cal_ci_lower": final_results["val_cal_ci"][0],
                    "val_cal_ci_upper": final_results["val_cal_ci"][1],
                    # Test metrics
                    "test_cindex_mean": final_results["test_cindex_mean"],
                    "test_cindex_std": final_results["test_cindex_std"],
                    "test_cindex_ci_lower": final_results["test_cindex_ci"][0],
                    "test_cindex_ci_upper": final_results["test_cindex_ci"][1],
                    "test_ibs_mean": final_results["test_ibs_mean"],
                    "test_ibs_std": final_results["test_ibs_std"],
                    "test_ibs_ci_lower": final_results["test_ibs_ci"][0],
                    "test_ibs_ci_upper": final_results["test_ibs_ci"][1],
                    "test_cal_mean": final_results["test_cal_mean"],
                    "test_cal_std": final_results["test_cal_std"],
                    "test_cal_ci_lower": final_results["test_cal_ci"][0],
                    "test_cal_ci_upper": final_results["test_cal_ci"][1],
                    # Metadata
                    "time_sec_total": optim_time,
                    "n_bootstrap_iters": cfg.n_bootstrap_iters,
                    "n_trials": cfg.n_trials,
                    "space_file": space_path,
                    "status": "ok",
                    "error": "",
                }

                logger.success(
                    f"[{dataset_name}] {model_name} COMPLETED:\n"
                    f"  Train: C-Index={final_results['train_cindex_mean']:.4f}±{final_results['train_cindex_std']:.4f}, "
                    f"IBS={final_results['train_ibs_mean']:.4f}±{final_results['train_ibs_std']:.4f}, "
                    f"Cal={final_results['train_cal_mean']:.4f}±{final_results['train_cal_std']:.4f}\n"
                    f"  Val:   C-Index={final_results['val_cindex_mean']:.4f}±{final_results['val_cindex_std']:.4f}, "
                    f"IBS={final_results['val_ibs_mean']:.4f}±{final_results['val_ibs_std']:.4f}, "
                    f"Cal={final_results['val_cal_mean']:.4f}±{final_results['val_cal_std']:.4f}\n"
                    f"  Test:  C-Index={final_results['test_cindex_mean']:.4f}±{final_results['test_cindex_std']:.4f}, "
                    f"IBS={final_results['test_ibs_mean']:.4f}±{final_results['test_ibs_std']:.4f}, "
                    f"Cal={final_results['test_cal_mean']:.4f}±{final_results['test_cal_std']:.4f}"
                )

                if cfg.save_final_model:
                    # Train and save final model on combined train+val set
                    logger.info(f"[{dataset_name}] Training to saving final model with best parameters on train+val set")
                    
                    # Create train+val split (exclude only test set)
                    n_total = len(df)
                    try:
                        stratify = df[cfg.event_col].astype(bool).to_numpy()
                        trainval_idx, test_idx = train_test_split(
                            np.arange(n_total),
                            test_size=cfg.test_size,
                            random_state=cfg.seed,
                            stratify=stratify,
                        )
                    except ValueError:
                        trainval_idx, test_idx = train_test_split(
                            np.arange(n_total),
                            test_size=cfg.test_size,
                            random_state=cfg.seed,
                        )

                    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
                    
                    # Preprocess
                    prep = TabularPreprocessor(
                        TabularPreprocessorConfig(event_col=cfg.event_col, time_col=cfg.time_col)
                    )
                    X_trainval = prep.fit_transform(df_trainval)
                    y_trainval = build_survival_y(df_trainval, event_col=cfg.event_col, time_col=cfg.time_col)
                    
                    # Train final model
                    final_model = make_estimator(model_name, best_params, cfg.interpolation)
                    
                    # Handle dvcsurv pre-training if needed
                    if "converse" in model_name or "dvcsurv" in model_name:
                        final_model.pretraining = True
                        final_model.warm_start = False
                        final_model.fit(X_trainval, y_trainval)
                        final_model.initialize_centers(X_trainval, batch_size=final_model.batch_size, random_state=cfg.seed)
                        final_model.pretraining = False
                        final_model.warm_start = True
                    
                    final_model.fit(X_trainval, y_trainval)
                    
                    # Save model and test data
                    model_dir = "saved_models"
                    os.makedirs(model_dir, exist_ok=True)
                    model_filename = f"{dataset_name}_{model_name}_seed{cfg.seed}.joblib"
                    model_path = os.path.join(model_dir, model_filename)
                    
                    # Get test split for saving
                    df_test = df.iloc[test_idx].reset_index(drop=True)
                    
                    joblib.dump({
                        'model': final_model,
                        'preprocessor': prep,
                        'best_params': best_params,
                        'dataset': dataset_name,
                        'model_name': model_name,
                        'seed': cfg.seed,
                        'test_idx': test_idx,  # Save test indices for reproducibility
                        'metrics': {
                            'test_cindex_mean': final_results['test_cindex_mean'],
                            'test_ibs_mean': final_results['test_ibs_mean'],
                            'test_cal_mean': final_results['test_cal_mean'],
                        }
                    }, model_path)
                    
                    # Save test data separately for easy access in notebooks
                    test_data_filename = f"{dataset_name}_test_seed{cfg.seed}.csv"
                    test_data_path = os.path.join(model_dir, test_data_filename)
                    df_test.to_csv(test_data_path, index=False)
                    
                    logger.info(f"[{dataset_name}] Final model saved to {model_path}")
                    logger.info(f"[{dataset_name}] Test data saved to {test_data_path}")


            except Exception as e:
                logger.exception(f"[{dataset_name}] Optimization failed for {model_name}: {e}")
                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": cfg.seed,
                    "best_params": json.dumps({}),
                    # Train metrics
                    "train_cindex_mean": float("nan"),
                    "train_cindex_std": float("nan"),
                    "train_cindex_ci_lower": float("nan"),
                    "train_cindex_ci_upper": float("nan"),
                    "train_ibs_mean": float("nan"),
                    "train_ibs_std": float("nan"),
                    "train_ibs_ci_lower": float("nan"),
                    "train_ibs_ci_upper": float("nan"),
                    "train_cal_mean": float("nan"),
                    "train_cal_std": float("nan"),
                    "train_cal_ci_lower": float("nan"),
                    "train_cal_ci_upper": float("nan"),
                    # Val metrics
                    "val_cindex_mean": float("nan"),
                    "val_cindex_std": float("nan"),
                    "val_cindex_ci_lower": float("nan"),
                    "val_cindex_ci_upper": float("nan"),
                    "val_ibs_mean": float("nan"),
                    "val_ibs_std": float("nan"),
                    "val_ibs_ci_lower": float("nan"),
                    "val_ibs_ci_upper": float("nan"),
                    "val_cal_mean": float("nan"),
                    "val_cal_std": float("nan"),
                    "val_cal_ci_lower": float("nan"),
                    "val_cal_ci_upper": float("nan"),
                    # Test metrics
                    "test_cindex_mean": float("nan"),
                    "test_cindex_std": float("nan"),
                    "test_cindex_ci_lower": float("nan"),
                    "test_cindex_ci_upper": float("nan"),
                    "test_ibs_mean": float("nan"),
                    "test_ibs_std": float("nan"),
                    "test_ibs_ci_lower": float("nan"),
                    "test_ibs_ci_upper": float("nan"),
                    "test_cal_mean": float("nan"),
                    "test_cal_std": float("nan"),
                    "test_cal_ci_lower": float("nan"),
                    "test_cal_ci_upper": float("nan"),
                    # Metadata
                    "time_sec_total": float("nan"),
                    "n_bootstrap_iters": cfg.n_bootstrap_iters,
                    "n_trials": cfg.n_trials,
                    "space_file": space_path,
                    "status": "failed",
                    "error": str(e),
                }

            pd.DataFrame([row]).to_csv(summary_path, index=False, mode="a", header=False)


if __name__ == "__main__":
    main()
