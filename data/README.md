# Benchmark scRNA-seq Datasets

## 1. Bundled Quickstart Demo Dataset (`data/demo/`)
- `data/demo/pbmc_demo_500cells.h5ad` (0.57 MB): Stratified 536-cell slice from the Human PBMC (PbmcBench / SCP424) atlas spanning all 9 immune lineages and 8 batches (7 sequencing protocols) over the 3,000 PBMC HVGs.
- Run `python3 code/demo_inference.py` to verify loading and inference.

## 2. Full Four-Atlas Processed Benchmarks (`data/real/processed/*.h5ad`)
Because the four full gzip-compressed `.h5ad` benchmark atlases exceed GitHub's 100 MB per-file Git push limit (total 1.89 GiB), they are not tracked directly in Git history:

| File | Atlas | Cells | Cell Types | Batches | Size | Public Primary Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `pancreas.h5ad` | Human Pancreas Islets | 14,229 | 14 | 4 | 257 MiB | GSE84133, GSE85241, E-MTAB-5061, GSE81608 |
| `pbmc.h5ad` | Human PBMC (PbmcBench) | 23,154 | 9 | 8 | 94 MiB | Broad Single Cell Portal SCP424 |
| `tabula_muris.h5ad` | Mouse Tabula Muris (12 Organs) | 78,241 | 74 | 2 | 986 MiB | GSE109774 / Figshare |
| `lung_covid.h5ad` | Human BALF COVID-19 | 65,813 | 10 | 13 | 550 MiB | GSE145926 & GSM3660650 (GSE128033) |

### Option A — Automated Download & Preprocessing from Public Repositories
```bash
python3 code/real_data/download_benchmarks.py --all
```

### Option B — Direct Preprocessed Download (GitHub Releases)
If the preprocessed `.h5ad` files are attached to the repository's **GitHub Releases (`v1.0`)**, place all four `.h5ad` files into `data/real/processed/` and verify integrity against `data/real/processed/dataset_checksums.sha256`:
```bash
cd data/real/processed && shasum -a 256 -c dataset_checksums.sha256
```
