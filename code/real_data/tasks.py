#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation protocol (task definitions). Everything is fixed a priori and does
not depend on any method's results.

1. intra   : 5-fold stratified CV inside each batch (study / protocol)
             (Abdelaal et al. 2019 intra-dataset protocol; populations with
             < 10 cells in that batch are removed, as in the benchmark).
2. cross   : train on reference batch(es), test on an unseen protocol/study.
             - pancreas: leave-one-study-out (train on 3 studies, test on the 4th)
             - pbmc    : train on 10Xv2 (pbmc1), test on every other protocol and
                         on 10Xv2 pbmc2 (Abdelaal PbmcBench protocol)
             Test cells whose label does not occur in training are kept in the
             prediction files but excluded from closed-set metrics.
3. openset : same reference/query split as one cross task per dataset, but one
             cell type is removed from training ("leave-one-cell-type-out",
             cf. Abdelaal et al. rejection experiment). Repeated for every cell
             type with >= MIN_OPEN_TRAIN cells in training and >= MIN_OPEN_TEST
             cells in the query.

A stratified 15 % validation subset of the training cells is provided for
early stopping / threshold calibration; no method may look at test labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import anndata as ad

MIN_INTRA_CELLS = 10
MIN_OPEN_TRAIN = 20
MIN_OPEN_TEST = 20
VAL_FRACTION = 0.15

# Reference / query designs for the open-set experiment (one per dataset).
# key = obs column used to split reference/query; test=None -> all other values.
OPENSET_DESIGN = {
    "pancreas": {"key": "batch", "train": ["Baron (inDrop)"], "test": None},
    "pbmc": {"key": "batch", "train": ["10Xv2 (pbmc1)"], "test": None},
    "tabula_muris": {"key": "batch", "train": ["FACS (Smart-seq2)"], "test": ["Droplet (10x)"]},
    "lung_covid": {"key": "condition", "train": ["Healthy", "Moderate"], "test": ["Severe"]},
}

# Column defining the groups inside which intra-dataset 5-fold CV is run
INTRA_GROUP = {"pancreas": "batch", "pbmc": "batch", "tabula_muris": "batch", "lung_covid": None}

# Natural (biologically motivated) open-set task: cell types that are absent/near-absent
# in the reference condition and expand in the query condition.
NATURAL_OPENSET = {"lung_covid": ["Neutrophil"]}


@dataclass
class Task:
    task_id: str
    dataset: str
    setting: str                     # intra | cross | openset
    label_key: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_batches: List[str]
    test_batches: List[str]
    heldout_types: List[str] = field(default_factory=list)
    fold: Optional[int] = None

    def describe(self) -> Dict[str, object]:
        return dict(
            task_id=self.task_id, dataset=self.dataset, setting=self.setting, label_key=self.label_key,
            n_train=len(self.train_idx), n_val=len(self.val_idx), n_test=len(self.test_idx),
            train_batches=";".join(self.train_batches), test_batches=";".join(self.test_batches),
            heldout_types=";".join(self.heldout_types), fold=self.fold,
        )


def _val_split(idx: np.ndarray, labels: np.ndarray, seed: int):
    lab = labels[idx]
    vc = pd.Series(lab).value_counts()
    strat = lab if (vc.min() >= 2) else None
    tr, va = train_test_split(idx, test_size=VAL_FRACTION, stratify=strat, random_state=seed)
    return np.sort(tr), np.sort(va)


def intra_tasks(adata: ad.AnnData, dataset: str, label_key: str = "cell_type_original",
                n_splits: int = 5, seed: int = 0) -> List[Task]:
    tasks = []
    labels_all = adata.obs[label_key].astype(str).to_numpy()
    gkey = INTRA_GROUP.get(dataset, "batch")
    batches = adata.obs[gkey].astype(str).to_numpy() if gkey else np.full(adata.n_obs, "all", dtype=object)
    for b in sorted(pd.unique(batches)):
        idx_b = np.where(batches == b)[0]
        vc = pd.Series(labels_all[idx_b]).value_counts()
        ok = set(vc[vc >= MIN_INTRA_CELLS].index)
        idx_b = idx_b[np.isin(labels_all[idx_b], list(ok))]
        if len(ok) < 2:
            continue
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (trv, te) in enumerate(skf.split(idx_b, labels_all[idx_b])):
            tr, va = _val_split(idx_b[trv], labels_all, seed + k)
            safe_b = b.replace(" ", "").replace("(", "_").replace(")", "")
            tasks.append(Task(f"{dataset}|intra|{safe_b}|fold{k}", dataset, "intra", label_key,
                              tr, va, np.sort(idx_b[te]), [b], [b], fold=k))
    return tasks


def cross_tasks(adata: ad.AnnData, dataset: str, label_key: str = "cell_type", seed: int = 0) -> List[Task]:
    labels_all = adata.obs[label_key].astype(str).to_numpy()
    batches = adata.obs["batch"].astype(str).to_numpy()
    tasks = []
    if dataset == "pancreas":
        for b in sorted(pd.unique(batches)):
            tr_all = np.where(batches != b)[0]
            te = np.where(batches == b)[0]
            tr, va = _val_split(tr_all, labels_all, seed)
            tasks.append(Task(f"{dataset}|cross|test={b}", dataset, "cross", label_key, tr, va, te,
                              sorted(set(batches[tr_all])), [b]))
    elif dataset == "pbmc":
        ref = "10Xv2 (pbmc1)"
        tr_all = np.where(batches == ref)[0]
        tr, va = _val_split(tr_all, labels_all, seed)
        for b in sorted(pd.unique(batches)):
            if b == ref:
                continue
            te = np.where(batches == b)[0]
            tasks.append(Task(f"{dataset}|cross|test={b}", dataset, "cross", label_key, tr, va, te, [ref], [b]))
    elif dataset == "tabula_muris":
        for ref, qry in [("FACS (Smart-seq2)", "Droplet (10x)"), ("Droplet (10x)", "FACS (Smart-seq2)")]:
            tr_all = np.where(batches == ref)[0]
            te = np.where(batches == qry)[0]
            tr, va = _val_split(tr_all, labels_all, seed)
            tasks.append(Task(f"{dataset}|cross|{ref}->{qry}", dataset, "cross", label_key, tr, va, te, [ref], [qry]))
    elif dataset == "lung_covid":
        cond = adata.obs["condition"].astype(str).to_numpy()
        tr_all = np.where(np.isin(cond, ["Healthy", "Moderate"]))[0]
        te = np.where(cond == "Severe")[0]
        tr, va = _val_split(tr_all, labels_all, seed)
        tasks.append(Task(f"{dataset}|cross|Healthy+Moderate->Severe", dataset, "cross", label_key, tr, va, te,
                          ["Healthy", "Moderate"], ["Severe"]))
    else:
        raise KeyError(dataset)
    return tasks


def openset_tasks(adata: ad.AnnData, dataset: str, label_key: str = "cell_type", seed: int = 0,
                  max_heldout: int = 10) -> List[Task]:
    design = OPENSET_DESIGN[dataset]
    labels_all = adata.obs[label_key].astype(str).to_numpy()
    batches = adata.obs[design["key"]].astype(str).to_numpy()
    train_b = design["train"]
    test_b = design["test"] or sorted(set(batches) - set(train_b))
    tr_all = np.where(np.isin(batches, train_b))[0]
    te_all = np.where(np.isin(batches, test_b))[0]
    train_types = set(labels_all[tr_all])
    tasks = []
    vc_tr = pd.Series(labels_all[tr_all]).value_counts()
    vc_te = pd.Series(labels_all[te_all]).value_counts()
    eligible = [
        ct for ct in sorted(train_types)
        if vc_tr.get(ct, 0) >= MIN_OPEN_TRAIN and vc_te.get(ct, 0) >= MIN_OPEN_TEST
    ]
    if max_heldout and len(eligible) > max_heldout:
        # When a multi-organ atlas (e.g. Tabula Muris) has >30 shared types, select the top-K most
        # abundant shared populations in the query set so open-set evaluation stays balanced across datasets.
        eligible = sorted(eligible, key=lambda c: (-int(vc_te.get(c, 0)), c))[:max_heldout]
        eligible = sorted(eligible)
    for ct in eligible:
        tr_keep = tr_all[labels_all[tr_all] != ct]
        tr, va = _val_split(tr_keep, labels_all, seed)
        # query: known types seen in training + the held-out type
        known = train_types - {ct}
        te = te_all[np.isin(labels_all[te_all], list(known | {ct}))]
        tasks.append(Task(f"{dataset}|openset|heldout={ct}", dataset, "openset", label_key, tr, va, te,
                          list(train_b), list(test_b), heldout_types=[ct]))
    # natural open-set: the held-out types are removed from the reference regardless of their count
    nat = NATURAL_OPENSET.get(dataset)
    if nat:
        tr_keep = tr_all[~np.isin(labels_all[tr_all], nat)]
        tr, va = _val_split(tr_keep, labels_all, seed)
        known = set(labels_all[tr_keep])
        te = te_all[np.isin(labels_all[te_all], list(known | set(nat)))]
        tasks.append(Task(f"{dataset}|openset|natural={'+'.join(nat)}", dataset, "openset", label_key, tr, va, te,
                          list(train_b), list(test_b), heldout_types=list(nat)))
    elif dataset == "tabula_muris":
        # In Tabula Muris FACS -> Droplet, 15 cell types (>=20 cells) exist in Droplet but are absent in FACS
        unseen_te = sorted([ct for ct, cnt in vc_te.items() if ct not in train_types and cnt >= MIN_OPEN_TEST])
        if unseen_te:
            tr, va = _val_split(tr_all, labels_all, seed)
            te = te_all[np.isin(labels_all[te_all], list(train_types | set(unseen_te)))]
            tasks.append(Task(f"{dataset}|openset|natural=Droplet_unique", dataset, "openset", label_key, tr, va, te,
                              list(train_b), list(test_b), heldout_types=unseen_te))
    return tasks


def all_tasks(adata: ad.AnnData, dataset: str, settings=("intra", "cross", "openset"), seed: int = 0) -> List[Task]:
    out: List[Task] = []
    if "intra" in settings:
        out += intra_tasks(adata, dataset, seed=seed)
    if "cross" in settings:
        out += cross_tasks(adata, dataset, seed=seed)
    if "openset" in settings:
        out += openset_tasks(adata, dataset, seed=seed)
    return out

