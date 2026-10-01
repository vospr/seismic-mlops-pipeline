# Seismic facies MLOps pipeline

Patch classification into **real facies labels** on the public F3 North Sea benchmark
(Alaudah et al., 2019), with a promotion gate, one served sklearn Pipeline, and CI that can fail.

## Quick start

```bash
git clone https://github.com/vospr/seismic-mlops-pipeline && cd seismic-mlops-pipeline
make demo      # needs uv (https://docs.astral.sh/uv/); installs from uv.lock, trains on the committed
               # fixture, promotes through the gate, starts the API, queries it
make test      # 27 acceptance tests
```

`make demo` log from this repo: [`docs/results/demo_fixture.log`](docs/results/demo_fixture.log).

## What it does

```
F3 patches (32x32, real facies label = centre pixel)
   -> one sklearn Pipeline: StandardScaler -> PCA(32) -> HistGradientBoostingClassifier
      fitted on train sections only; gate decides on a validation block; test1 reported once
   -> logged to MLflow as a single model
   -> promotion gate: must beat majority-class baseline AND the current `champion` alias
   -> FastAPI loads models:/seismic-facies@champion and serves it as-is
```

- **Labels**: the six F3 facies from the benchmark. No synthetic labels.
- **Split** (`block_split` in `data.py`): the benchmark `train` volume is cut into contiguous blocks of sections:
  train = sections 0-290, a 10-section gap, validation = sections 301-400. Random patch splits would leak because
  overlapping patches of one section land on both sides. `test1` is the benchmark's separate volume.
- **Test1 is not used for any decision.** Model-family choice (`scripts/model_family_comparison.py`) and the gate
  use validation only; test1 is predicted once per run, after the gate decision, and only reported.
  Tests: `test_06_selection_leak.py` (gate result unchanged when test1 labels are scrambled; one test1 prediction per
  run; disjoint blocks; the comparison script never reads test1).
- **One model object**: `src/seismic_mlops/serve.py` has no feature code; it flattens the patch and calls the
  registered Pipeline. Test `test_04_serving.py` asserts served outputs equal the training Pipeline's and that
  the serving module contains no PCA/scaler/statistics code.
- **Gate** (`src/seismic_mlops/gate.py`): challenger must beat the majority-class baseline on accuracy and
  macro-F1, and the current champion on macro-F1 (re-scored on the same validation set). A rejected challenger is
  still registered as a version (auditable) but does not get the `champion` alias.

## Results (real benchmark, whatever they are)

Full benchmark: 20,000 train / 5,000 validation / 5,000 test1 patches
([`docs/results/f3_run1.log`](docs/results/f3_run1.log)). The gate decided on the validation column.

| | validation accuracy | validation macro-F1 | **test1 accuracy** | **test1 macro-F1** |
|---|---|---|---|---|
| Majority-class baseline | 0.524 | 0.137 | 0.515 | 0.113 |
| This model | 0.838 | 0.426 | **0.717** | **0.514** |

Test1 per class (same log): good on the large facies (class 2: F1 0.83; class 1: 0.80), but **class 4 recall is
0.09 and class 5 recall is 0.02**. Class 4 is 3.0% of sampled train patches and 15.6% of sampled test1 patches,
so this is partly a distribution shift between regions. This is a 32x32 patch classifier on raw pixels, not an
interpretation tool.

Second identical run was **rejected** by the gate ("macro_f1 0.426 does not beat champion 0.426",
[`docs/results/f3_run2.log`](docs/results/f3_run2.log)).

Committed fixture (600 train / 300 validation / 300 test1 patches) is much smaller and noisier. Validation 0.760
accuracy vs 0.570 baseline; test1 0.630 vs 0.543 accuracy, macro-F1 0.374 vs 0.117
([`docs/results/demo_fixture.log`](docs/results/demo_fixture.log)).

### Caveats

- **The model family was first looked at on test1, before the validation split existed.** An earlier version of this
  repo chose gradient boosting after seeing test1 numbers (logistic regression failed to beat the baseline there). The
  choice was then re-made on validation only and came out the same
  ([`docs/results/model_family_comparison.log`](docs/results/model_family_comparison.log): HGB 0.824 vs RF 0.788 vs
  logistic regression 0.507 validation accuracy, baseline 0.531). The earlier look cannot be undone, so treat test1 as
  mostly, not perfectly, untouched. No hyperparameter search was done.
- **The validation block is easy and nearly lacks rare facies**: class 3 is 0.7%, class 4 0.0% and class 5 0.2% of
  validation patches, so the gate's macro-F1 says little about rare-class performance. Validation accuracy (0.838)
  is higher than test1 accuracy (0.717) for the same reason.
- Patches are sampled at random positions inside each block, so patches within a block overlap spatially.
- Single seed, no confidence intervals.

## Get the real benchmark

Source: Alaudah, Michałowicz, Alfarraj, AlRegib, *A machine learning benchmark for facies classification*,
Interpretation 2019 (arXiv:1901.07659). Zenodo record 3755060.

```bash
make f3-download   # downloads data.zip (1,051,449,986 bytes), verifies MD5, unzips to data/f3/
make f3            # trains + gates on the full benchmark
```

- URL: `https://zenodo.org/record/3755060/files/data.zip`
- MD5: `bc5932279831a95c0b244fd765376d85` (matches Zenodo's published checksum; verified on the file used here)

`make fixture` regenerates `tests/fixtures/f3_patches.npz` (about 1.7 MB) from the downloaded benchmark with a fixed seed.

## Tests = acceptance checks

Written before the code they check (`tests/`):

| Check | Test |
|---|---|
| Real facies labels, fixture provenance, MD5 rejects corrupt file | `test_01_data.py` |
| One Pipeline; scaler/PCA fit on train only (no leakage) | `test_02_pipeline.py` |
| Gate rejects below-baseline, worse, and tied challengers; champion alias does not move | `test_03_gate.py` |
| No selection on test1: disjoint blocks, gate independent of test1, one test1 prediction per run | `test_06_selection_leak.py` |
| Served predictions == training Pipeline; bad shape rejected; no feature code in serve | `test_04_serving.py` |
| CI has no `continue-on-error`; deps are pyproject + lockfile; cut-scope components absent from code | `test_05_repo_hygiene.py` |

CI (`.github/workflows/ci.yml`) runs `make ci` = `pytest` + `make demo` on push to `main` and on PRs.
No step is allowed to fail silently.

## How it extends

Removed from this version to keep the pipeline connected end to end. Reasonable next steps, not implemented here:

- a feature store (Feast) once more than one model consumes the same features
- retrieval (FAISS/RAG) over run reports
- separate dev/staging/prod environments
- drift monitoring against the training reference, and API request metrics
- a 2D CNN on the patches in place of PCA + trees
