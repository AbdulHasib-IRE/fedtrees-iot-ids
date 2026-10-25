# fedtrees-iot-ids

Federated gradient-boosted decision trees (GBDT) for IoT intrusion detection on CIC IoT-DIAD 2024, compared with
federated neural detectors on identical clients, splits and seeds.

Three federated tree protocols are implemented and evaluated:

| Protocol | What is exchanged | File |
|---|---|---|
| One-shot ensemble | each client trains a full LightGBM model; the server averages class probabilities | `fedtree_core.py` (`run_ens`) |
| Cyclic boosting | one shared booster visits the clients in turn; each turn adds boosting iterations | `fedtree_core.py` (`run_cyc`) |
| Histogram aggregation | federated quantile binning, then summed per-bin gradient/Hessian histograms | `fedtree_hist.py` (`run_hist`) |

Histogram aggregation is simulated exactly: training LightGBM on the pooled data mapped to the federated bin indices
(one LightGBM bin per index, checked by an assertion in every run) grows the same trees as the summed histograms.

## Contents
| Path | Contents |
|---|---|
| `FedTrees.ipynb` | Colab notebook that ran every experiment (outputs cleared). Cell 1: setup and reproduction checks; Cell 2: one-shot ensemble and cyclic boosting, 6 settings x 10 seeds; Cell 3: diagnosis; Cell 4: histogram aggregation; Cell 6: settings checks and 50 clients; Cells 5 and 7: export. |
| `fedtree_core.py`, `fedtree_hist.py` | The federated tree protocols (LightGBM 4.6.0). |
| `compute_statistics.py` | Recomputes every reported statistic (means, paired tests with Holm correction, standard metrics and confusion matrices from the saved predictions, client composition, 50-client sizes, communication model and transfer times). |
| `results/grid/` | 120 runs: one-shot ensemble and cyclic boosting (JSON with history and metrics, plus test predictions as `.npy`). |
| `results/grid_hist/` | 60 histogram-aggregation runs (with bins per feature and leaves per tree) and 2 binning checks. |
| `results/grid_extra/` | 90 runs: cyclic boosting with 5 and 10 iterations per turn, ensemble with 100 iterations (source-IP clients), and 50-client runs; plus the hist-module reproduction check. |
| `results/diag/`, `results/check/` | Per-client class counts, test labels, and reproduction checks. |
| `benchmark/` | Federated FA-MLP runs on the same clients and seeds (10 and 50 clients), centralized references and the split summary, taken from the public `fedids-leakfree-benchmark` release (https://doi.org/10.5281/zenodo.23483716). |

Settings: 6 split-partition settings (random split: IID, Dirichlet 0.5, Dirichlet 0.1, source IP; time-block split:
Dirichlet 0.1, source IP), 10 clients, seeds 42, 123, 456, 789, 1011, 2024, 7, 99, 314, 2718. LightGBM: 300 iterations,
learning rate 0.05, 63 leaves, balanced class weights from aggregate class counts.

## Reproduce the statistics (no GPU, about 2 min)
    pip install -r requirements.txt
    python3 compute_statistics.py results benchmark benchmark benchmark statistics.json

## Re-run the experiments
The notebook reads the CIC IoT-DIAD 2024 splits, client partitions and federated neural harness of the public
`fedids-leakfree-benchmark` release (https://doi.org/10.5281/zenodo.23483716) from Google Drive. CPU runtime is enough.

## Data
CIC IoT-DIAD 2024 is not redistributed; it is available from the Canadian Institute for Cybersecurity
(https://www.unb.ca/cic/datasets/iot-diad-2024.html). The `.npy` label files hold only class indices of the test rows.

## License and citation
Code: MIT. Result files: CC BY 4.0. See `CITATION.cff`.

## Contact
Saurav Chandra Das (corresponding author), saurav0001@uftb.ac.bd
