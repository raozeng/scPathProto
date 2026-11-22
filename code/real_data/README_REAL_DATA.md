# Benchmark Datasets and Evaluation Protocol

## 1. Datasets

All four benchmark atlases use original count matrices and author-provided cell-type annotations from public repositories:

| Dataset | Source | Batches / Protocols | Cells × Genes | Cell Types |
| :--- | :--- | :--- | :---: | :--- |
| `pancreas` | Abdelaal et al. *Genome Biology* 2019 ([Zenodo 3357167](https://doi.org/10.5281/zenodo.3357167)): Baron (`GSE84133`), Muraro (`GSE85241`), Segerstolpe (`E-MTAB-5061`), Xin (`GSE81608`) | 4 studies / 4 platforms (inDrop, CEL-Seq2, Smart-Seq2, SMARTer) | 14,229 × 15,582 (gene intersection) | 14 harmonized types |
| `pbmc` | Ding et al. *Nat Biotechnol* 2020 PbmcBench ([Zenodo 3357167](https://doi.org/10.5281/zenodo.3357167)) | `pbmc1` across 7 protocols (10Xv2, 10Xv3, CEL-Seq2, Drop-Seq, inDrop, Seq-Well, Smart-Seq2) + `10Xv2 (pbmc2)` (8 batches) | 23,154 × 22,280 | 9 consensus types |
| `tabula_muris` | Tabula Muris Consortium *Nature* 2018 ([`GSE109774`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774), [`czbiohub-sf/tabula-muris`](https://github.com/czbiohub-sf/tabula-muris)) | 12 shared organs profiled by `FACS (Smart-seq2)` (23,376 cells) and `Droplet (10x)` (54,865 cells) | 78,241 × 23,341 (non-ERCC intersection) | 74 ontology classes (36 shared across both platforms) |
| `lung_covid` | Liao et al. *Nature Medicine* 2020 ([`GSE145926`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145926)) + HC4 (`GSM3660650`, `GSE128033`) + [`zhangzlab/covid_balf`](https://github.com/zhangzlab/covid_balf) | 13 donors across 3 severity groups: Healthy (HC1–HC4, 21,883), Moderate (M1–M3, 7,311), Severe (S1–S6, 36,619) | 65,813 × 33,538 | 10 author cell types |

Per-dataset cell-type-by-batch count tables are saved in `data/real/processed/*_celltype_by_batch.csv`.

**Pancreas label harmonization** (`loaders.PANCREAS_LABEL_MAP`) merges synonymous labels while preserving the original strings in `obs['cell_type_original']`:
- `duct` / `ductal` -> `ductal`
- `pp` / `gamma` -> `PP (gamma)`
- `activated_stellate` / `quiescent_stellate` / `mesenchymal` / `PSC` -> `stellate`
- Non-cell-type QC labels (`co-expression`, `unclassified endocrine`, etc.) are excluded.

**Count representation**: Muraro (UMI-corrected CEL-Seq2) and Xin (SMARTer) provide non-integer expression matrices. Values are rounded to integers inside `scANVI` only (negative binomial likelihood); all other methods use the original values.

## 2. Evaluation Protocol (`tasks.py`)

1. **Intra-dataset (`intra`)**: 5-fold stratified cross-validation within each batch. Populations with `< 10` cells in a batch are excluded following Abdelaal et al. (2019).
2. **Cross-platform / cross-condition (`cross`)**:
   - `pancreas`: leave-one-study-out across the 4 studies.
   - `pbmc`: train on `10Xv2 (pbmc1)`, test on each of the other 7 batches.
   - `tabula_muris`: bidirectional `FACS (Smart-seq2) <-> Droplet (10x)`.
   - `lung_covid`: `Healthy + Moderate -> Severe`.
   Test cells whose labels do not occur in the training set are excluded from closed-set metrics and counted in `n_excluded_unseen_label`.
3. **Open-set (`openset`)**:
   - **Leave-one-cell-type-out**: hold out one shared cell type (`>= 20` cells in both reference and query; up to the 10 most abundant shared types per atlas) from training and evaluate known-class accuracy and novel-cell detection.
   - **Natural biological open-set**: `Neutrophil` expansion in Severe COVID-19 (`lung_covid`) and 15 Droplet-specific lineages absent from FACS (`tabula_muris`).

**Split hygiene**:
- HVGs are selected on training cells only.
- A stratified 15% subset of training cells is held out as a validation split for early stopping and open-set threshold calibration (5th percentile of validation known-cell confidence, i.e., 95% known acceptance).
- Rare cell types are defined by training prevalence `< 2.5%`.

## 3. Baseline Wrappers (`baselines.py`)

| Method | Implementation | Input Features |
| :--- | :--- | :--- |
| SingleR | `singler` (BiocPy port, default parameters) | All genes (internal marker selection) |
| CellTypist | `celltypist.train(feature_selection=True)` + `annotate` | All genes (log1p, normalized to 1e4) |
| scANVI | `scvi-tools` SCVI -> SCANVI + scArches query surgery | Training HVGs (counts) |
| TOSICA | `code/third_party/TOSICA` (commit in `COMMIT.txt`) | Training HVGs (log1p) + dataset GMT |
| SVM | `CalibratedClassifierCV(LinearSVC())` (`SVM_rejection`) | All genes |
| scPathProto | Adapter to `code/model.py` | Training HVGs + same dataset GMT as TOSICA |

**Pathway priors (`preprocess.DATASET_GMT`)**:
- `pancreas`: `GO_bp.gmt`
- `pbmc`, `lung_covid`: `immune.gmt`
- `tabula_muris`: `m_GO_bp.gmt`

## 4. Running the Pipeline

```bash
# 1. Download raw benchmark files
python3 code/real_data/download_benchmarks.py
python3 code/real_data/download_geo.py

# 2. Build processed .h5ad files
python3 code/real_data/loaders.py

# 3. Run benchmark tasks (resumable)
python3 code/real_data/run_benchmark.py \
    --datasets pancreas pbmc tabula_muris lung_covid \
    --settings intra cross openset \
    --methods scPathProto SVM CellTypist SingleR scANVI TOSICA \
    --include_ablation
```
