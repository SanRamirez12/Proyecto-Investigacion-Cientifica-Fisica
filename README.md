# Gamma-ray Source Classification with Artificial Neural Networks

This repository contains the complete implementation of a machine learning pipeline for the classification of gamma-ray sources detected by the **Fermi Large Area Telescope (Fermi-LAT)**.  
The project focuses on identifying **Active Galactic Nuclei (AGNs)** using data from the **4FGL-DR4 catalog**, with special emphasis on robust generalization to unlabeled and Galactic source populations.

---

## Project Overview

The main goal of this project is to **automatically classify gamma-ray sources** into three astrophysically meaningful categories:

- **FSRQ** (Flat Spectrum Radio Quasars)  
- **BLL** (BL Lacertae objects)  
- **NoAGN** (non–active galactic nucleus sources)

A **Multi-Layer Perceptron (MLP)** artificial neural network was designed, optimized, and validated to address this task, integrating both astrophysical domain knowledge and modern machine learning practices.

This work was developed as the **final research project for the Physics undergraduate program**, combining expertise from **Physics** and **Computer Systems Engineering**.

---

## Scientific Context

Gamma-ray source catalogs such as **4FGL-DR4** contain thousands of detected sources, many of which remain **unassociated or ambiguously classified**.  
Manual classification is time-consuming and limited by observational constraints, motivating the use of **data-driven classification methods**.

This project contributes a **validated ANN-based methodology** for source classification and uncertainty reduction in large astrophysical catalogs.

---

## Methodology

The project follows the **CRISP-ML (Cross-Industry Standard Process for Machine Learning)** framework, adapted for astrophysical research:

1. **Astrophysical understanding** of gamma-ray catalogs and AGN populations  
2. **Data engineering**: cleaning, preprocessing, exploratory data analysis (EDA)  
3. **Feature engineering** based on spectral and variability parameters  
4. **Model engineering**: ANN design and optimization  
5. **Evaluation and scientific validation**

---

## Model Description

- **Architecture**: Multi-Layer Perceptron (MLP)
- **Inputs**: 15 physically motivated spectral and variability features from 4FGL-DR4
- **Output**: Multiclass probability (FSRQ / BLL / NoAGN)
- **Class imbalance handling**: SMOTENC
- **Hyperparameter optimization**: Optuna (Bayesian optimization, 410 trials)
- **Frameworks**: TensorFlow / Keras, scikit-learn

---

## Results

- **Accuracy**: **87.77%**
- **Weighted F1-score**: **87.75%**
- **Stable generalization** across:
  - Blazar Candidates of Uncertain Type (BCU)
  - Galactic sources in the **Vela supernova remnant region**
- Successfully identified **non-extragalactic populations**, confirming robustness beyond training data.

---

## Research Integration

This ML-based classification pipeline and validation workflow directly contributed to the research project:

> **“GeV emission in the region of the Vela supernova remnant: a new view of the shell”**

- Research group led by **Dr. Miguel Araya**
- Currently **under review** in *Astronomy & Astrophysics (A&A)*

The methodology developed here supports the interpretation of unidentified Fermi-LAT sources and the study of extended gamma-ray emission.

---

## Project Structure

The repository is organized following a clear separation of concerns, aligned with the CRISP-ML methodology and typical research-grade machine learning workflows.


Proyecto-Investigacion-Cientifica-Fisica/
│
├── data/
│   ├── raw/                  # Original data extracted from the 4FGL-DR4 catalog
│   ├── external/             # External reference datasets or auxiliary catalogs
│   ├── interim/              # Intermediate datasets generated during preprocessing
│   └── processed/            # Final curated datasets used for training and evaluation
│
├── src/
│   ├── data exploration/
│   │   ├── data_exploration.py        # Exploratory Data Analysis (EDA) and diagnostics
│   │   ├── utils_data_exp.py           # Utility functions for EDA and preprocessing
│   │   └── vela_sources_preprocessing.py
│   │                                   # Preprocessing pipeline for Vela region sources
│   │
│   ├── model development/
│   │   ├── hyperparameter_optuna.py    # Bayesian hyperparameter optimization with Optuna
│   │   ├── model_3classes_pipeline.py  # ANN pipeline for 3-class classification (FSRQ/BLL/NoAGN)
│   │   ├── model_4classes_pipeline.py  # Extended ANN pipeline including additional AGN classes
│   │   ├── training_final_montecarlo_cv.py
│   │                                   # Final Monte Carlo training with stratified cross-validation
│   │   ├── utils_hyperparameter_opt.py # Helper utilities for optimization workflows
│   │   └── utils_model_dev.py           # Model construction, compilation, and training utilities
│   │
│   └── random tests/
│       ├── PruebasRandom.py             # Early experimental scripts and sanity checks
│       ├── pruebas_Pandas.py            # Pandas-based data inspection tests
│       ├── pruebas_iniciales.py         # Initial prototype experiments
│       ├── codigo_random.py             # Miscellaneous exploratory code
│       └── expAnalysis_bases.py          # Early experimental analysis foundations
│
├── plots/
│   ├── training_curves/        # Loss and metric evolution plots
│   ├── confusion_matrices/    # Confusion matrices for trained models
│   └── eda_figures/            # Figures generated during exploratory data analysis
│
├── README.md                   # Project documentation and scientific context
├── .gitignore                  # Git ignore rules (caches, IDE configs, non-versionable files)
└── requirements.txt            # Python dependencies for reproducibility

### Structure Design Notes

- All folder names are **lowercase and consistent**, avoiding cross-platform issues between Windows and Linux systems.
- The `src/` directory is structured by **functional responsibility** rather than execution order.
- Experimental and legacy scripts are isolated under `random tests/` to keep the main pipeline clean.
- The structure supports **reproducibility**, **paper traceability**, and future extension of the pipeline.

---

## Tools & Dependencies

- **Python** 3.10+
- **Core libraries**:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow / keras
  - astropy
  - optuna
  - imbalanced-learn
  - livelossplot
  - joblib

---

## Project Status

 **Completed**  
 **Research output submitted to peer-reviewed journal**

---

##  Author & Collaboration

**Santiago Ramírez Elizondo**  
Physics & Computer Systems Engineering  
Universidad de Costa Rica

In collaboration with:
- **Dr. Miguel Araya**
- **Diego Bueso**
- **Dr. Braulio Solano**

---

## License

This project is released for academic and research purposes.  
Please cite appropriately if used in scientific work.

