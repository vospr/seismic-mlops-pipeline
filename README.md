# Seismic facies MLOps pipeline

Patch classification into **real facies labels** on the public F3 North Sea benchmark
(Alaudah et al., 2019), with a promotion gate, one served sklearn Pipeline, and CI that can fail.

## Quick start

```bash
git clone https://github.com/vospr/seismic-mlops-pipeline && cd seismic-mlops-pipeline
make demo      # needs uv (https://docs.astral.sh/uv/); installs from uv.lock, trains on the committed
               # fixture, promotes through the gate, starts the API, queries it
make test      # 21 acceptance tests
```

`make demo` log from this repo: [`docs/results/demo_fixture.log`](docs/results/demo_fixture.log).

## What it does

```
F3 patches (32x32, real facies label = centre pixel)
   -> one sklearn Pipeline: StandardScaler -> PCA(32) -> HistGradientBoostingClassifier
      fitted on the train volume only, scored on a disjoint test volume
   -> logged to MLflow as a single model
   -> promotion gate: must beat majority-class baseline AND the current `champion` alias
   -> FastAPI loads models:/seismic-facies@champion and serves it as-is
```

- **Labels**: the six F3 facies from the benchmark. No synthetic labels.
- **Split**: train on the benchmark `train` volume, evaluate on the `test1` volume (spatially disjoint).
- **One model object**: `src/seismic_mlops/serve.py` has no feature code; it flattens the patch and calls the
  registered Pipeline. Test `test_04_serving.py` asserts served outputs equal the training Pipeline's and that
  the serving module contains no PCA/scaler/statistics code.
- **Gate** (`src/seismic_mlops/gate.py`): challenger must beat the majority-class baseline on accuracy and
  macro-F1, and the current champion on macro-F1 (re-scored on the same test set). A rejected challenger is
  still registered as a version (auditable) but does not get the `champion` alias.

## Results (real benchmark, whatever they are)

Full benchmark, 20,000 train patches / 5,000 held-out test1 patches
([`docs/results/f3_run1.log`](docs/results/f3_run1.log)):

| | accuracy | macro-F1 |
|---|---|---|
| Majority-class baseline | 0.515 | 0.113 |
| This model | **0.705** | **0.489** |

Per class ([`docs/results/f3_per_class.txt`](docs/results/f3_per_class.txt)): good on the large facies
(class 2: F1 0.83; class 1: 0.77), but **class 4 recall is 0.06 and class 5 is never detected**. Class 4 is
2.3% of the sampled train patches but 15.6% of the sampled test1 patches, so this is partly a train/test
distribution shift. This is a 32x32 patch classifier on raw pixels, not an interpretation tool.

Second identical run was **rejected** by the gate ("macro_f1 0.489 does not beat champion 0.489",
[`docs/results/f3_run2.log`](docs/results/f3_run2.log)).

Committed fixture (600 train / 300 test patches) is much smaller and noisier: accuracy 0.627 vs baseline 0.543,
macro-F1 0.380 vs 0.117 ([`docs/results/demo_fixture.log`](docs/results/demo_fixture.log)).

### Caveats

- **Model family was chosen after looking at test1.** Logistic regression on the same Pipeline did *not* beat the
  baseline on test1 (accuracy 0.503 vs 0.543 on the fixture; 0.504 vs 0.511 on a 6,000-patch sample;
  [`docs/results/model_family_comparison.log`](docs/results/model_family_comparison.log)), so
  `HistGradientBoostingClassifier` was used. No hyperparameter search was done. The gate also evaluates on test1
  on every run, so test1 is not a one-shot test set (the benchmark intends `test_once` to be used once).
- Patches are sampled at random positions, so neighbouring train patches overlap spatially.
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
