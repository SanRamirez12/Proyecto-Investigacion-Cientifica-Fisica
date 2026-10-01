# Gamma-ray Source Classification with Artificial Neural Networks

Multi-layer perceptron (MLP) that classifies *Fermi*-LAT **4FGL-DR4** point sources into **BLL**, **FSRQ** and **NoAGN**, using only spectral-shape and variability parameters from the catalogue. The model was applied to Blazar Candidates of Uncertain type (BCU) and to unassociated sources in the **Vela supernova remnant** region. Those Vela results are part of a peer-reviewed publication:

> Araya, M., Ramírez, S., Bueso, D. & Solano-Rojas, B. J. (2026).
> **GeV emission in the region of Vela: A new view of the supernova remnant.**
> *Astronomy & Astrophysics*, 710, A74. https://doi.org/10.1051/0004-6361/202557331

Undergraduate research project, Escuela de Física, Universidad de Costa Rica (2025).

---

## Key results

| | Value |
|---|---|
| Training set | 3,064 sources (1,490 BLL · 819 FSRQ · 755 NoAGN) |
| Inputs | 15 catalogue parameters → 17 columns (`SpectrumType` one-hot encoded) |
| Stratified shuffle CV (10 splits) | Accuracy **85.4 ± 1.4 %** · weighted F1 **0.854 ± 0.014** · per-class AUC > 0.94 |
| Selected final model (held-out 20 %, n = 613) | Accuracy 87.8 % · weighted F1 0.878 |
| BCU (n = 1,623) | 828 BLL · 566 FSRQ · 229 NoAGN |
| Vela region (35 sources, paper) | No source reaches P(BLL) or P(FSRQ) ≥ 0.70 |

> The cross-validation figure is the most reliable estimate of generalization. The final-model test split was also used for hyperparameter selection (see *Known limitations*).

---

## Pipeline

```
gll_psc_v35.fit ──► data exploration ──► data/post preliminary analysis/*.parquet
                                              │
                ┌─────────────────────────────┼──────────────────────────┐
                ▼                             ▼                          ▼
     hyperparameter_optuna.py     model_3classes_pipeline.py   training_final_montecarlo_cv.py
     (Optuna, 411 trials)          (10-split stratified CV)     (100 runs → best model .h5/.pkl)
                                                                         │
                                              ┌──────────────────────────┴──────────┐
                                              ▼                                     ▼
                                     bcu_evaluation.py                     vela_evaluation.py
```

| Step | Script | Output |
|---|---|---|
| 1. Read catalogue, relabel classes, one-hot `SpectrumType`, drop NaN/inf rows | `src/data exploration/data_exploration.py` | `data/post preliminary analysis/df_final_*.parquet` |
| 2. Extract Vela-region sources by name | `src/data exploration/vela_sources_preprocessing.py` | `fuentes_vela*.parquet` |
| 3. Hyperparameter search (TPE + MedianPruner, weighted-F1 objective) | `src/model development/hyperparameter_optuna.py` | `data/hyperparameter studies/` |
| 4. Stratified shuffle cross-validation with SMOTENC | `src/model development/model_3classes_pipeline.py` | `data/fold results training/` |
| 5. Final training (100 runs, metric thresholds, best model kept) | `src/model development/training_final_montecarlo_cv.py` | `data/monte carlo results/` |
| 6. Apply to BCU / Vela | `src/model evaluation/*.py` | `data/model evaluation/` |

Run each script from its own folder (scripts use paths relative to their location).

### Final architecture
4 hidden layers `[121, 105, 137, 80]` · activations `relu, selu, gelu, selu` · dropout `[0.25, 0.15, 0.10, 0.10]` · softmax output (3) · AdamW (lr = 8.53 × 10⁻⁴) · batch 55 · early stopping (patience 50) · sparse categorical cross-entropy · class imbalance handled with SMOTENC.

---

## Data

| Folder | Contents |
|---|---|
| `data/raw/` | `gll_psc_v35.fit` (4FGL-DR4); 4LAC-DR2 and 3PC catalogues used in early exploration |
| `data/post preliminary analysis/` | Cleaned datasets: full, 3-class training set, BCU, unassociated, Vela variants |
| `data/hyperparameter studies/` | Pickled Optuna studies and top-N trial tables |
| `data/fold results training/` | Per-fold models, histories and reports |
| `data/monte carlo results/` | Final model (`.h5`, `.pkl` with scaler) and metrics CSV |
| `data/model evaluation/` | Per-source class probabilities for Vela |
| `plots/` | EDA, learning curves, confusion matrices, Optuna diagnostics |

Class mapping from `CLASS1`: `fsrq`→FSRQ, `bll`→BLL, `bcu`→BCU, `rdg/nlsy1/sey/agn/css/ssrq`→OtroAGN (excluded, 81 sources), empty→unassociated, everything else→NoAGN.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas pyarrow astropy scikit-learn imbalanced-learn \
            tensorflow optuna==4.3.* plotly matplotlib seaborn livelossplot joblib
```

`optuna==4.3.*` is required to unpickle the stored studies. `numpy<2` is required by `leer_fits()` (uses `ndarray.newbyteorder`).

---

## Known limitations

- The final-model test split (`random_state=42`) is identical to the Optuna validation split, and the best of 100 runs is selected on it, so 87.8 % is optimistic.
- In the final training, `validation_split=0.2` is applied after SMOTENC without shuffling, so the synthetic samples end up in the validation slice.
- `src/feature engineering/` is a placeholder (empty).

---

## Repository structure

```
data/      raw catalogues, processed datasets and model artefacts
plots/     figures used in the thesis, presentation and paper
src/
  data exploration/     reading, cleaning, EDA, Vela extraction
  model development/    Optuna search, cross-validation, final training
  model evaluation/     BCU and Vela inference
  feature engineering/  (placeholder)
  random tests/         early prototypes
```

---

## Authors

**Santiago Ramírez Elizondo**: Physics & Computer Systems Engineering, Universidad de Costa Rica

Advisors and collaborators: Dr. Miguel Araya Arguedas (Escuela de Física, UCR) · MSc. Braulio Solano Rojas (ECCI, UCR) · Diego Bueso (UCR)

## Citation

```bibtex
@article{Araya2026Vela,
  author  = {Araya, Miguel and Ram{\'i}rez, Santiago and Bueso, Diego and Solano-Rojas, Braulio J.},
  title   = {GeV emission in the region of Vela: A new view of the supernova remnant},
  journal = {Astronomy \& Astrophysics},
  volume  = {710},
  pages   = {A74},
  year    = {2026},
  doi     = {10.1051/0004-6361/202557331}
}
```

## License
Released for academic and research use. Please cite the paper above if you use this work.
