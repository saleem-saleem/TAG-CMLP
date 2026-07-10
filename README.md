# TAG-CMLP
A Unified Temporal Graph-Based and Uncertainty-Aware Framework for Early Academic Performance Prediction
eference implementation for reproducing the experiments in the paper.
The code implements the three stages of the framework and the full evaluation
protocol (repeated cross-validation, confidence intervals, significance testing,
and a Friedman/Nemenyi critical-difference analysis).


---

## Framework overview

| Stage | Module | Input | Output |
|------|--------|-------|--------|
| I  | **TFEM** – Temporal Feature Evolution Modeling | raw per-student temporal sequences `x_i^t` | adaptive temporal embeddings `z_i` |
| II | **GAIL** – Graph-based Adaptive Influence Learning | the embeddings `{z_i}` (a similarity graph is built from them) | graph-refined embeddings + base-learner probabilities |
| III| **CUDM** – Confidence-aware Uncertainty Decision Module | the ensemble probability distribution | Cognitive Risk Index, selectively corrected prediction, reliability score |

Only **TFEM** touches the raw data; **GAIL** operates on the embeddings and
**CUDM** on the ensemble probabilities.

Key formulations (equation numbers refer to the paper):
- Temporal Academic Evolution Score `TAES` (Eq. 8) and Temporal Stability `TS` (Eq. 9)
- Adaptive representation `z_i = h_i + lambda1*TAES_i - lambda2*TS_i` (Eq. 10)
- Peer Learning Influence Score `PLIS` (Eq. 18)
- Reliability weighting `R_k` (Eq. 21) with ECE calibration and predictive variance
- Cognitive Risk Index `CRI` (Eq. 27) and the reliability metric `REL` (Eq. 30)

---

## Installation

```bash
git clone https://github.com/<user>/tag-cmlp.git
cd tag-cmlp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the Open University Learning Analytics Dataset (OULAD) from
<https://analyse.kmi.open.ac.uk/open_dataset> and unzip the seven CSV tables into
`data/oulad/`. If that folder is absent, the loader generates a small **synthetic**
dataset with the same schema so the pipeline runs end-to-end.

```
data/oulad/
  studentInfo.csv  studentRegistration.csv  studentVle.csv  vle.csv
  studentAssessment.csv  assessments.csv  courses.csv
```

## Reproducing the experiments

```bash
# Early-prediction experiment (weekly cutoffs, all models)
python -m scripts.run_early_prediction --task pass_fail    --config config.yaml
python -m scripts.run_early_prediction --task pass_withdrawn --config config.yaml

# Significance testing + critical-difference diagram (from saved per-run scores)
python -m scripts.run_significance --results results/pass_fail_runs.csv
```

Outputs (tables of mean +/- s.d. with 95% CIs, the significance table, and the
CD diagram) are written to `results/`.

## Repository layout

```
tag-cmlp/
  config.yaml
  requirements.txt
  src/
    data/    oulad_loader.py  feature_engineering.py
    models/  tfem.py  gail.py  cudm.py  tag_cmlp.py
    baselines.py  metrics.py  evaluation.py  utils.py
  scripts/ run_early_prediction.py  run_significance.py
```



## Notes on reproducibility & testing

- The data pipeline, GAIL, CUDM, metrics, and the full statistical evaluation
  (cross-validation, confidence intervals, Wilcoxon/Holm, Friedman/Nemenyi, and the
  critical-difference diagram) are pure NumPy/scikit-learn/SciPy and run anywhere.
- **TFEM** and the neural sequence baselines (GRU/LSTM/CNN-LSTM) require a working
  PyTorch install (`pip install torch`). Use a CPU or CUDA build appropriate to your
  machine.
- **GCN/GAT/MTGNN** are included as dependency-light proxies so the pipeline runs
  out of the box; replace them in `src/baselines.py` with full graph implementations
  (e.g. PyTorch Geometric) to reproduce those baselines exactly.

