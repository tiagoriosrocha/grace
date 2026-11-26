# GRACE: relational alignment of claims and evidence

Code, converted graph inputs and saved predictions supporting the paper
**GRACE: Graph-based Relational Alignment of Claims and Evidence for Fact Verification**.

## Layout

| Path                            | Contents                                                                                                                    |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `experiments/comparison/`       | Exact comparison code: GRACE, KGAT-style and GMN-style; FactKG also contains the published TF-IDF and Serialized BERT runs. |
| `experiments/ablation/`         | Exact final ablation code.                                                                                                  |
| `experiments/legacy_baselines/` | Historical code used for TF-IDF and Serialized BERT on HoVer, FEVER and FEVEROUS.                                           |
| `data/`                         | Four exact input exports.                                                                                                   |
| `results/`                      | 23 selected original run bundles, with predictions, metrics, splits, configurations and manifests.                          |
| `analysis/`                     | New publication utilities to verify files and recompute the reported metrics without training.                              |
| `paper_reference/`              | Original numeric LaTeX tables and the active probability-metric figure from the manuscript.                                 |
| `provenance/`                   | Input and source hashes, selected runs, artifact inventory and checksums.                                                   |
| `docs/`                         | Reproduction details, mapping of source projects and remaining limits.                                                      |

The paper uses `text_tfidf` (TF-IDF), `graph_pair_transformer` (Serialized BERT),
`kgat_style`, `gmn_style` and `relational_gnn` (GRACE). Other systems remain in
the byte-identical historical sources and original result bundles. They are not
additional systems reported in the current manuscript.

## Verify the archive and reproduce tables without training

Python 3.10+ is sufficient for the archive checks and point metrics; no PyTorch or
network connection is required:

```sh
python3 analysis/verify_archive.py
python3 analysis/recompute_tables.py
```

Derived CSVs and the verification report are written to `analysis/generated/`.
For paired group bootstrap intervals, install `numpy==1.26.4` in Python 3.10–3.12,
then run:

```sh
python3 analysis/recompute_tables.py --bootstrap 2000
```

This option resamples complete test groups using `split_seed + 9001`, shares each
draw across systems and seeds, and averages seed-specific Macro-F1. The intervals
are exploratory and pointwise, conditional on the saved models and split.

## Retrain

The archived environment used Linux, Python 3.12.3, PyTorch 2.7.1+cu128 and
torch-geometric 2.6.1. Exact package versions are recorded in every run manifest.
The original RTX 5080 launcher installs its environment and requires CUDA:

```sh
cd experiments/comparison
./RODAR_5080.sh --dataset factkg --systems text_tfidf,graph_pair_transformer,kgat_style,gmn_style,relational_gnn
./RODAR_5080.sh --dataset hover --systems kgat_style,gmn_style,relational_gnn
./RODAR_5080.sh --dataset fever --systems kgat_style,gmn_style,relational_gnn
./RODAR_5080.sh --dataset feverous --systems kgat_style,gmn_style,relational_gnn
```

Retrain the historical baseline paths on the other three datasets separately:

```sh
cd experiments/legacy_baselines
./RODAR_5080.sh --dataset hover --systems text_tfidf,graph_pair_transformer
./RODAR_5080.sh --dataset fever --systems text_tfidf,graph_pair_transformer
./RODAR_5080.sh --dataset feverous --systems text_tfidf,graph_pair_transformer
```

Each block above starts from the repository root. The supplied snapshots contain
the original runners and their broader model ordering. Running a subset can alter
the random-number-generator state relative to the original full-system run. To
repeat the exact original selection/order for each run, use its saved `config.json`
`systems` list in `provenance/runs.json`'s referenced result directory. Numeric
identity after retraining is not guaranteed across hardware or software changes.

The four published ablations disable one component for time:

```sh
cd experiments/ablation
./RODAR_5080.sh --dataset factkg --ablation no-null
./RODAR_5080.sh --dataset factkg --ablation no-relational-context
./RODAR_5080.sh --dataset factkg --ablation no-nli
./RODAR_5080.sh --dataset factkg --ablation no-condition-aggregation
```

Repeat this block for `hover`, `fever` and `feverous`. `--ablation full` is available
for a GRACE-only reference run. The published full-model reference is taken from
the corresponding comparison run; the identical FactKG ablation reference was
deduplicated. The seeds are 7, 21 and 42; the threshold is 0.5. Fresh runs get new
directories inside each snapshot's `results/`, leaving the archived runs unchanged.

Models are downloaded on first use; no API key is required for training on the
included converted inputs. CPU use requires a compatible manually prepared
environment and `python run.py --device cpu`; the RTX launcher itself requires CUDA.

After installing the ML environment, the following publication utility reuses a
selected run's exact saved configuration, including its original system order.
Run it from the repository root with the Python of that environment:

```sh
python analysis/run_recorded.py --run article_20261112_100731_338092 --device cuda
python analysis/run_recorded.py --run article_20261112_100731_338092 --device cpu --audit-only
```

Repeat for the names in `provenance/runs.json`. It creates a fresh output directory
and does not reuse or replace archived runs. The helper has been syntax-checked;
ML training and GPU execution were not rerun during this packaging task.

## Scope and availability

The archive supports verification of the saved predictions and reexecution from
the supplied graph exports. These are grouped internal partitions, not official
benchmark test scores. KGAT-style and GMN-style are protocol-specific adaptations.

Pretrained weights, large feature caches and fitted checkpoints are not bundled.
Most selected run directories do not contain neural checkpoints locally; the
per-run inventory distinguishes unavailable files from deliberately omitted ones.
Graph extraction prompts, exact extraction code/version and generation logs have
not yet been tied to these exports. Consequently regeneration from raw benchmark
texts is not established by this package. See `docs/REPRODUCIBILITY.md`.

Dataset terms, third-party notices and a project license must be settled before
public redistribution. This preparation does not assign a new license to the data
or copied code. `docs/DATA_DISTRIBUTION.md` describes separating large data assets
from the code repository without changing their names or hashes.
