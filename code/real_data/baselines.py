#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrappers around baseline annotation methods and scPathProto.

| Name           | Implementation                                                 | Reference |
|----------------|----------------------------------------------------------------|-----------|
| SingleR        | `singler` (BiocPy port, default settings)                      | Aran et al., Nat Immunol 2019 |
| CellTypist     | `celltypist.train` / `celltypist.annotate`                     | Domínguez Conde et al., Science 2022 |
| scANVI         | `scvi-tools` SCVI -> SCANVI + scArches query surgery           | Xu et al., Mol Syst Biol 2021; Lotfollahi et al., Nat Biotechnol 2022 |
| TOSICA         | `code/third_party/TOSICA` (commit in COMMIT.txt)               | Chen et al., Nat Commun 2023 |
| SVM            | `CalibratedClassifierCV(LinearSVC())` (`SVM_rejection`)        | Abdelaal et al., Genome Biol 2019 |

Each wrapper implements ``fit(adata_train, label_key)`` and ``predict(adata_test)``.
Inputs are AnnData objects with log1p-normalized ``X`` and raw counts in
``layers['counts']``. ``predict`` returns a dict with ``labels``, ``confidence``,
and ``proba`` (DataFrame or None).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

import anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
TOSICA_DIR = os.path.abspath(os.path.join(HERE, "..", "third_party", "TOSICA"))


def _dense(X) -> np.ndarray:
    return X.toarray() if sp.issparse(X) else np.asarray(X)


class BaseAnnotator:
    name = "base"
    outputs_probability = True

    def __init__(self, seed: int = 0, n_jobs: int = 8, **kwargs):
        self.seed = seed
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.features: Optional[List[str]] = None

    def set_features(self, features: Optional[List[str]]):
        """Optional gene list (selected on training cells by the runner)."""
        self.features = list(features) if features is not None else None
        return self

    def fit(self, adata_train: ad.AnnData, label_key: str):
        raise NotImplementedError

    def predict(self, adata_test: ad.AnnData) -> Dict[str, object]:
        raise NotImplementedError

    @staticmethod
    def _pack(labels, confidence, proba: Optional[pd.DataFrame]) -> Dict[str, object]:
        return {
            "labels": np.asarray(labels).astype(str),
            "confidence": np.asarray(confidence, dtype=np.float64),
            "proba": proba,
        }


# -----------------------------------------------------------------------------
# SVM (Abdelaal et al. 2019 reference implementation)
# -----------------------------------------------------------------------------
class SVMAnnotator(BaseAnnotator):
    """Linear SVM with Platt-calibrated probabilities (Abdelaal 'SVM_rejection')."""

    name = "SVM"

    def fit(self, adata_train, label_key):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC

        genes = self.features or list(adata_train.var_names)
        self.genes_ = genes
        X = adata_train[:, genes].X
        y = adata_train.obs[label_key].astype(str).to_numpy()
        vc = pd.Series(y).value_counts()
        cv = int(min(3, vc.min())) if vc.min() >= 2 else None
        base = LinearSVC(random_state=self.seed)
        self.clf_ = CalibratedClassifierCV(base, cv=cv) if cv and cv >= 2 else base
        self.clf_.fit(X, y)
        return self

    def predict(self, adata_test):
        X = adata_test[:, self.genes_].X
        if hasattr(self.clf_, "predict_proba"):
            P = self.clf_.predict_proba(X)
            proba = pd.DataFrame(P, columns=self.clf_.classes_, index=adata_test.obs_names)
            lab = self.clf_.classes_[P.argmax(1)]
            return self._pack(lab, P.max(1), proba)
        d = self.clf_.decision_function(X)
        return self._pack(self.clf_.predict(X), d.max(1), None)


# -----------------------------------------------------------------------------
# CellTypist (official)
# -----------------------------------------------------------------------------
class CellTypistAnnotator(BaseAnnotator):
    name = "CellTypist"

    def fit(self, adata_train, label_key):
        import celltypist

        a = adata_train if self.features is None else adata_train[:, self.features]
        a = ad.AnnData(X=a.X.copy(), obs=a.obs[[label_key]].copy(), var=pd.DataFrame(index=a.var_names))
        # Official recommended workflow for custom models (CellTypist docs / tutorial):
        # log1p-normalised (10,000 counts) input, two-pass feature selection.
        self.model_ = celltypist.train(
            a, labels=label_key, n_jobs=self.n_jobs, feature_selection=True,
            use_SGD=self.kwargs.get("use_SGD", False), check_expression=self.features is None,
        )
        return self

    def predict(self, adata_test):
        import celltypist

        a = ad.AnnData(X=adata_test.X.copy(), var=pd.DataFrame(index=adata_test.var_names),
                       obs=pd.DataFrame(index=adata_test.obs_names))
        res = celltypist.annotate(a, model=self.model_, majority_voting=False, mode="best match")
        P = res.probability_matrix.loc[a.obs_names]
        lab = res.predicted_labels.loc[a.obs_names, "predicted_labels"].astype(str).to_numpy()
        return self._pack(lab, P.max(axis=1).to_numpy(), P)


# -----------------------------------------------------------------------------
# SingleR (BiocPy port, official algorithm)
# -----------------------------------------------------------------------------
class SingleRAnnotator(BaseAnnotator):
    name = "SingleR"
    outputs_probability = False

    def fit(self, adata_train, label_key):
        self.ref_ = adata_train if self.features is None else adata_train[:, self.features]
        self.ref_labels_ = self.ref_.obs[label_key].astype(str).to_list()
        return self

    def predict(self, adata_test):
        import singler

        ref = self.ref_
        common = [g for g in ref.var_names if g in set(adata_test.var_names)]
        ref_mat = sp.csc_matrix(ref[:, common].X).T.tocsr().astype(np.float64)   # genes x cells
        test_mat = sp.csc_matrix(adata_test[:, common].X).T.tocsr().astype(np.float64)
        built = singler.train_single(ref_mat, self.ref_labels_, ref_features=common,
                                     test_features=common, num_threads=self.n_jobs)
        res = singler.classify_single(test_mat, built, num_threads=self.n_jobs)
        best = np.asarray(res.get_column("best")).astype(str)
        scores_bf = res.get_column("scores")
        score_cols = list(scores_bf.get_column_names())
        S = np.column_stack([np.asarray(scores_bf.get_column(c), dtype=float) for c in score_cols])
        scores = pd.DataFrame(S, columns=score_cols, index=adata_test.obs_names)
        out = self._pack(best, S.max(1), None)
        out["scores"] = scores
        try:
            out["delta"] = np.asarray(res.get_column("delta"), dtype=float)
        except Exception:  # noqa: BLE001
            pass
        return out


# -----------------------------------------------------------------------------
# scANVI (official scvi-tools; scArches surgery for the query)
# -----------------------------------------------------------------------------
class ScanviAnnotator(BaseAnnotator):
    name = "scANVI"

    @staticmethod
    def _integer_counts(X):
        """scVI/scANVI model counts with a negative binomial likelihood. Some studies
        (e.g. Muraro et al., CEL-Seq2) distribute UMI-corrected non-integer values;
        these are rounded for scANVI only (all other methods use the original values)."""
        X = sp.csr_matrix(X, dtype=np.float32, copy=True)
        if X.nnz and np.any(np.abs(X.data - np.round(X.data)) > 1e-6):
            X.data = np.round(X.data)
            X.eliminate_zeros()
        return X

    def fit(self, adata_train, label_key):
        import scvi

        scvi.settings.seed = self.seed
        self.label_key = label_key
        genes = self.features or list(adata_train.var_names)
        self.genes_ = genes
        ref = ad.AnnData(X=self._integer_counts(adata_train[:, genes].layers["counts"]),
                         obs=adata_train.obs[[label_key, "batch"]].copy(),
                         var=pd.DataFrame(index=genes))
        ref.obs[label_key] = ref.obs[label_key].astype(str)
        ref.obs["batch"] = ref.obs["batch"].astype(str)
        self.ref_batches_ = set(ref.obs["batch"].unique())
        ref.layers["counts"] = ref.X.copy()
        scvi.model.SCVI.setup_anndata(ref, layer="counts", batch_key="batch", labels_key=label_key)
        # architecture recommended for scArches reference mapping (scvi-tools tutorial)
        arch = dict(n_layers=2, n_latent=30, encode_covariates=True, deeply_inject_covariates=False,
                    use_layer_norm="both", use_batch_norm="none")
        vae = scvi.model.SCVI(ref, **arch)
        tr_kw = dict(accelerator=self.kwargs.get("accelerator", "auto"), enable_progress_bar=False)
        vae.train(max_epochs=self.kwargs.get("scvi_epochs"), **tr_kw)  # None -> scvi default heuristic
        self.lvae_ = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category="Unknown", labels_key=label_key)
        self.lvae_.train(max_epochs=self.kwargs.get("scanvi_epochs", 20), n_samples_per_label=100, **tr_kw)
        self.tr_kw_ = tr_kw
        return self

    def predict(self, adata_test):
        import scvi

        missing = [g for g in self.genes_ if g not in set(adata_test.var_names)]
        if missing:
            raise ValueError(f"{len(missing)} reference genes missing from query")
        q_batches = set(adata_test.obs["batch"].astype(str).unique())
        q = ad.AnnData(X=self._integer_counts(adata_test[:, self.genes_].layers["counts"]),
                       obs=pd.DataFrame({"batch": adata_test.obs["batch"].astype(str).to_numpy(),
                                         self.label_key: "Unknown"}, index=adata_test.obs_names),
                       var=pd.DataFrame(index=self.genes_))
        q.layers["counts"] = q.X.copy()
        if q_batches.issubset(self.ref_batches_):
            # Validation / intra-dataset cells belong to reference batches already seen during training
            P = self.lvae_.predict(q, soft=True)
        else:
            # Unseen query batches: run official scArches query surgery
            scvi.model.SCANVI.prepare_query_anndata(q, self.lvae_)
            qm = scvi.model.SCANVI.load_query_data(q, self.lvae_)
            qm.train(max_epochs=self.kwargs.get("query_epochs", 100), plan_kwargs=dict(weight_decay=0.0),
                     check_val_every_n_epoch=10, **self.tr_kw_)
            P = qm.predict(soft=True)
        P = P.loc[q.obs_names] if isinstance(P, pd.DataFrame) else pd.DataFrame(P, index=q.obs_names)
        P = P[[c for c in P.columns if c != "Unknown"]]
        lab = P.columns.to_numpy()[P.to_numpy().argmax(1)]
        return self._pack(lab, P.max(axis=1).to_numpy(), P)


# -----------------------------------------------------------------------------
# TOSICA (official code, default hyper-parameters)
# -----------------------------------------------------------------------------
class TosicaAnnotator(BaseAnnotator):
    name = "TOSICA"

    def fit(self, adata_train, label_key):
        if TOSICA_DIR not in sys.path:
            sys.path.insert(0, TOSICA_DIR)
        import TOSICA  # noqa: F401
        from preprocess import GMT_FILES

        self.workdir_ = tempfile.mkdtemp(prefix="tosica_")
        genes = self.features or list(adata_train.var_names)
        self.genes_ = genes
        ref = ad.AnnData(X=_dense(adata_train[:, genes].X).astype(np.float32),
                         obs=adata_train.obs[[label_key]].astype(str).copy(), var=pd.DataFrame(index=genes))
        self.epochs_ = int(self.kwargs.get("tosica_epochs", 10))
        bsize = int(self.kwargs.get("tosica_batch_size", 64))
        lr_scaled = float(self.kwargs.get("tosica_lr", 0.001))
        cwd = os.getcwd()
        try:
            os.chdir(self.workdir_)
            gmt = GMT_FILES[self.kwargs.get("gmt", "human_gobp")]
            TOSICA.train(ref, gmt_path=gmt, project="proj", label_name=label_key,
                         epochs=self.epochs_, batch_size=bsize, lr=lr_scaled)
        finally:
            os.chdir(cwd)
        return self

    def predict(self, adata_test):
        import TOSICA

        q = ad.AnnData(X=_dense(adata_test[:, self.genes_].X).astype(np.float32),
                       obs=pd.DataFrame(index=adata_test.obs_names), var=pd.DataFrame(index=self.genes_))
        cwd = os.getcwd()
        try:
            os.chdir(self.workdir_)
            w = os.path.join(self.workdir_, "proj", f"model-{self.epochs_ - 1}.pth")
            new = TOSICA.pre(q, model_weight_path=w, project="proj", cutoff=0.0, batch_size=256)
        finally:
            os.chdir(cwd)
        lab = new.obs["Prediction"].astype(str).to_numpy()
        conf = new.obs["Probability"].astype(float).to_numpy()
        return self._pack(lab, conf, None)

    def __del__(self):
        wd = getattr(self, "workdir_", None)
        if wd and os.path.isdir(wd):
            shutil.rmtree(wd, ignore_errors=True)


# -----------------------------------------------------------------------------
# scPathProto (this work) -- adapter to code/model.py
# -----------------------------------------------------------------------------
class ScPathProtoAnnotator(BaseAnnotator):
    """
    Adapter that feeds real data into ``ScPathProtoModel`` (code/model.py).
    Gene features = training-set HVGs; pathway prior = same GMT as TOSICA (preprocess.DATASET_GMT).
    """

    name = "scPathProto"
    ablation_flags: Dict[str, bool] = {}

    def fit(self, adata_train, label_key):
        code_dir = os.path.abspath(os.path.join(HERE, ".."))
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from model import ScPathProtoConfig, ScPathProtoModel
        from preprocess import GMT_FILES, build_pathway_mask

        genes = self.features or list(adata_train.var_names)
        self.genes_ = genes
        self.all_genes_ = list(adata_train.var_names)
        M, pnames = build_pathway_mask(genes, gmt_path=GMT_FILES[self.kwargs.get("gmt", "human_gobp")],
                                       max_pathways=int(self.kwargs.get("max_pathways", 300)))
        y = adata_train.obs[label_key].astype(str).to_numpy()
        self.classes_ = np.array(sorted(set(y)))
        cmap = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([cmap[c] for c in y])
        b = adata_train.obs["batch"].astype(str).to_numpy()
        self.batch_map_ = {v: i for i, v in enumerate(sorted(set(b)))}
        bi = np.array([self.batch_map_[v] for v in b])
        X = _dense(adata_train[:, genes].X).astype(np.float32)
        X_full = adata_train.X.tocsr() if sp.issparse(adata_train.X) else sp.csr_matrix(adata_train.X)
        flags = dict(
            use_zirg=bool(self.kwargs.get("use_zirg", self.ablation_flags.get("use_zirg", True))),
            use_mpca=bool(self.kwargs.get("use_mpca", self.ablation_flags.get("use_mpca", True))),
            use_hasgmp=bool(self.kwargs.get("use_hasgmp", self.ablation_flags.get("use_hasgmp", True))),
            use_proto_align=bool(self.kwargs.get("use_proto_align", self.ablation_flags.get("use_proto_align", True))),
            use_evidential=bool(self.kwargs.get("use_evidential", self.ablation_flags.get("use_evidential", True))),
        )
        cfg = ScPathProtoConfig(
            n_genes=len(genes), n_classes=len(self.classes_), n_batches=len(self.batch_map_),
            n_known_pathways=M.shape[0], seed=self.seed, **flags
        )
        self.model_ = ScPathProtoModel(
            config=cfg, pathway_mask=M, pathway_names=pnames, gene_names=genes, all_gene_names=self.all_genes_
        )
        self.model_.fit(X_train=X, y_train=yi, batch_train=bi, X_full=X_full)
        return self

    def predict(self, adata_test):
        X = _dense(adata_test[:, self.genes_].X).astype(np.float32)
        X_full_raw = adata_test[:, self.all_genes_].X if getattr(self, "all_genes_", None) is not None else None
        X_full = (
            (X_full_raw.tocsr() if sp.issparse(X_full_raw) else sp.csr_matrix(X_full_raw))
            if X_full_raw is not None
            else None
        )
        b = adata_test.obs["batch"].astype(str).to_numpy()
        nxt = len(self.batch_map_)
        bmap = dict(self.batch_map_)
        for v in sorted(set(b)):
            if v not in bmap:
                bmap[v] = nxt
                nxt += 1
        bi = np.array([bmap[v] for v in b])
        out = self.model_.predict_detailed(X, bi, X_full=X_full)
        P = pd.DataFrame(out["probs"], columns=self.classes_, index=adata_test.obs_names)
        lab = self.classes_[out["pred_labels"]]
        res = self._pack(lab, P.max(axis=1).to_numpy(), P)
        if "novelty_score" in out:
            res["confidence"] = -np.asarray(out["novelty_score"], dtype=float)
        return res


class ScPathProtoNoZIRG(ScPathProtoAnnotator):
    name = "scPathProto_wo_ZIRG"
    ablation_flags = {"use_zirg": False}


class ScPathProtoNoMPCA(ScPathProtoAnnotator):
    name = "scPathProto_wo_MPCA"
    ablation_flags = {"use_mpca": False}


class ScPathProtoNoHASGMP(ScPathProtoAnnotator):
    name = "scPathProto_wo_HASGMP"
    ablation_flags = {"use_hasgmp": False}


class ScPathProtoNoProtoAlign(ScPathProtoAnnotator):
    name = "scPathProto_wo_ProtoAlign"
    ablation_flags = {"use_proto_align": False}


class ScPathProtoNoEvidential(ScPathProtoAnnotator):
    name = "scPathProto_wo_Evidential"
    ablation_flags = {"use_evidential": False}


ABLATION_METHODS = {
    "scPathProto_wo_ZIRG": ScPathProtoNoZIRG,
    "scPathProto_wo_MPCA": ScPathProtoNoMPCA,
    "scPathProto_wo_HASGMP": ScPathProtoNoHASGMP,
    "scPathProto_wo_ProtoAlign": ScPathProtoNoProtoAlign,
    "scPathProto_wo_Evidential": ScPathProtoNoEvidential,
}

METHODS = {
    "SVM": SVMAnnotator,
    "CellTypist": CellTypistAnnotator,
    "SingleR": SingleRAnnotator,
    "scANVI": ScanviAnnotator,
    "TOSICA": TosicaAnnotator,
    "scPathProto": ScPathProtoAnnotator,
    **ABLATION_METHODS,
}

BASE_METHODS = ["SVM", "CellTypist", "SingleR", "scANVI", "TOSICA", "scPathProto"]

# Feature set given to each method (selected on training cells by the runner):
#   "all"  -> all genes shared by train and test (method applies its own default selection)
#   "hvg"  -> top HVGs of the training cells
FEATURE_POLICY = {
    "SVM": "all",           # Abdelaal et al. used all genes
    "CellTypist": "all",    # CellTypist performs its own feature selection
    "SingleR": "all",       # SingleR performs its own marker selection
    "scANVI": "hvg",        # scvi-tools tutorials: HVG-restricted counts
    "TOSICA": "hvg",        # dense input; HVGs keep memory tractable
    "scPathProto": "hvg",
    **{k: "hvg" for k in ABLATION_METHODS},
}
