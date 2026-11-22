#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation metrics.

Closed set (intra / cross)
    Test cells whose true label never occurs in the training set cannot be
    predicted by any closed-set classifier; following Abdelaal et al. (2019)
    they are excluded from closed-set metrics (their number is reported).
    Rare cell types are defined from the TRAINING set (prevalence < 2.5 %).

Open set
    novelty score = -confidence (each method's own confidence output).
    AUROC / AUPRC: held-out (novel) cells are the positive class.
    The rejection threshold is calibrated WITHOUT test labels: it is the
    5 % quantile of the confidences of the method on the validation cells
    (known types only), i.e. 95 % of known validation cells are accepted.
    H-score = harmonic mean of known-class accuracy (rejections count as
    errors) and novel-class recall (fraction of novel cells rejected)
    (Fu et al., ECCV 2020 definition).
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

RARE_PREVALENCE = 0.025


def rare_classes_from_train(y_train: Sequence[str], thr: float = RARE_PREVALENCE) -> list:
    vc = pd.Series(np.asarray(y_train).astype(str)).value_counts(normalize=True)
    return sorted(vc[vc < thr].index.tolist())


def closed_set_metrics(y_true, y_pred, train_classes: Iterable[str], rare: Iterable[str]) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    train_classes = set(train_classes)
    m = np.isin(y_true, list(train_classes))
    yt, yp = y_true[m], y_pred[m]
    labels = sorted(set(yt))
    out: Dict[str, float] = {
        "n_eval": int(m.sum()),
        "n_excluded_unseen_label": int((~m).sum()),
    }
    if len(yt) == 0:
        return out
    f1s = f1_score(yt, yp, labels=labels, average=None, zero_division=0)
    out.update(
        accuracy=accuracy_score(yt, yp),
        macro_f1=float(np.mean(f1s)),
        median_f1=float(np.median(f1s)),
        weighted_f1=f1_score(yt, yp, labels=labels, average="weighted", zero_division=0),
        balanced_accuracy=balanced_accuracy_score(yt, yp),
        cohen_kappa=cohen_kappa_score(yt, yp),
    )
    rare_present = [c for c in rare if c in labels]
    out["n_rare_types"] = len(rare_present)
    out["rare_f1"] = float(np.mean([f1s[labels.index(c)] for c in rare_present])) if rare_present else np.nan
    return out


def per_class_f1(y_true, y_pred, train_classes: Iterable[str]) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    m = np.isin(y_true, list(set(train_classes)))
    labels = sorted(set(y_true[m]))
    f1s = f1_score(y_true[m], y_pred[m], labels=labels, average=None, zero_division=0)
    sup = pd.Series(y_true[m]).value_counts()
    return pd.DataFrame({"cell_type": labels, "f1": f1s, "support": [int(sup[l]) for l in labels]})


def expected_calibration_error(conf, correct, n_bins: int = 15) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(conf) == 0 or np.any(conf < 0) or np.any(conf > 1 + 1e-9):
        return np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def open_set_metrics(
    y_true_test, y_pred_test, conf_test, heldout: Iterable[str],
    conf_val_known, accept_rate: float = 0.95, has_probability: bool = True,
) -> Dict[str, float]:
    y_true = np.asarray(y_true_test).astype(str)
    y_pred = np.asarray(y_pred_test).astype(str)
    conf = np.asarray(conf_test, dtype=float)
    novel = np.isin(y_true, list(set(heldout)))
    out: Dict[str, float] = {"n_known_test": int((~novel).sum()), "n_novel_test": int(novel.sum())}
    if novel.sum() == 0 or (~novel).sum() == 0:
        return out
    score = -conf
    out["auroc"] = roc_auc_score(novel, score)
    out["auprc"] = average_precision_score(novel, score)
    tau = float(np.quantile(np.asarray(conf_val_known, dtype=float), 1.0 - accept_rate))
    reject = conf < tau
    known_correct = (~reject[~novel]) & (y_pred[~novel] == y_true[~novel])
    acc_known = float(known_correct.mean())
    rec_novel = float(reject[novel].mean())
    prec_novel = float(novel[reject].mean()) if reject.any() else 0.0
    f1_novel = 2 * prec_novel * rec_novel / (prec_novel + rec_novel) if (prec_novel + rec_novel) > 0 else 0.0
    out.update(
        threshold=tau,
        acc_known_closed=float((y_pred[~novel] == y_true[~novel]).mean()),
        acc_known=acc_known,
        novel_recall=rec_novel,
        novel_precision=prec_novel,
        novel_f1=f1_novel,
        h_score=2 * acc_known * rec_novel / (acc_known + rec_novel) if (acc_known + rec_novel) > 0 else 0.0,
    )
    if has_probability:
        out["ece_known"] = expected_calibration_error(conf[~novel], (y_pred[~novel] == y_true[~novel]))
    return out

