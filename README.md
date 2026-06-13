# Identification of Breast Cancer Biomarkers using Machine Learning

This project identifies robust gene expression signatures for breast cancer and provides a machine learning pipeline capable of distinguishing tumor from normal tissue across different experimental platforms (GSE42568/GPL570 and GSE15852/GPL96).

## 🚀 Key Features
- **Informatics-Standard Pipeline:** Uses Scikit-Learn `Pipeline` to wrap scaling, feature selection, and classification, preventing data leakage.
- **Robust Statistical Metrics:** Includes Balanced Accuracy, MCC, Brier Score, and Calibration Curve/Slope diagnostics.
- **Cross-Platform Validation:** Achieved an **ROC-AUC of 0.91** (Discovery) and **0.89** (External Validation).
- **Large-Scale Generalizability:** Validated in **TCGA-BRCA (n=1,098)** and **METABRIC (n=1,904)** cohorts.
- **Clinical & Survival Analysis:** Includes Multivariate Cox regression, VIF diagnostics, Subtype-adjusted analysis, and Decision Curve Analysis (DCA).
- **Orthogonal Immune Deconvolution:** Cross-validated via **ESTIMATE** (stromal/immune scores) and **TIMER** (core immune lineages).
- **Model Interpretability:** Includes **SHAP (SHapley Additive exPlanations)** analysis for transparency.
- **Decision Curve Analysis (DCA):** Now includes **95% Confidence Intervals** for clinical utility assessment.

## 👥 Authors & Supervision
- **Huzaifa Akmal** (Lead Developer/Author) - Sanata Dharma University
- **Narendar Kumar** (Supervisor) - University of Technology of Belfort-Montbéliard (UTBM)

## 📁 Repository Structure
- `notebooks/`: Comprehensive pipeline from preprocessing to validation.
- `models/`: Finalized `breast_cancer_pipeline_v1.pkl` for diagnostic inference.
- `figures/`: High-resolution publication-ready visualizations.
- `requirements.txt`: Environment specification for full reproducibility.
- `LICENSE`: MIT License terms.
- `Data/`: Directory structure for datasets (GSE42568, GSE15852).

## 📁 Supplementary Code & Reproducibility
The following notebooks constitute the **Supplementary Code** for the study:
1. `01_Informatics_Standard_ML_Pipeline.ipynb`: Core model training and calibration.
2. `04_Model_Benchmarking_and_Comparison.ipynb`: Performance comparisons, SHAP analysis, and DCA.
3. `05_Biological_Validation.ipynb`: GO/KEGG enrichment and ESTIMATE/TIMER validation.
4. `08_Batch_Effect_and_Subtype_Analysis.ipynb`: PVCA and subtype-adjusted survival.

All code is documented and designed for reproducibility of the study findings.

## 🛠️ Usage
### 1. Installation
Install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Running the Analysis
Run the Jupyter notebooks in order (`00` through `08`) to reproduce the study.

### 3. Inference
To use the trained model for new predictions:
```python
import joblib

# Load the full pipeline
pipeline = joblib.load("models/breast_cancer_pipeline_v1.pkl")

# Predict on a new sample (DataFrame with common gene symbols)
prediction = pipeline.predict(new_data)
```

## 📊 Results
- **Final Model:** Balanced Linear SVM (5-gene Signature)
- **Primary Biomarkers:** FHL1, PCK1, ACVR1C, FABP4, MT1H.
- **External Validation ROC-AUC:** 0.89.
- **Prognostic Impact:** Signature score correlates with significantly improved relapse-free survival (HR = 0.64), though captured primarily by clinical grade.
- **Immune Context:** Higher signature expression is a surrogate for CD8+ T cell infiltration ('immune-hot' state).

## 📜 Citation & DOI
This work is archived on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20679726.svg)](https://doi.org/10.5281/zenodo.20679726)

All methodology follows standard bioinformatics best practices and TRIPOD-AI reporting guidelines.
