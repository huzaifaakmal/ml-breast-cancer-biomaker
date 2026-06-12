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

## 📁 Supplementary Code & Reproducibility
The following notebooks constitute the **Supplementary Code** for the manuscript:
1. `01_Informatics_Standard_ML_Pipeline.ipynb`: Core model training and calibration.
2. `04_Model_Benchmarking_and_Comparison.ipynb`: Performance comparisons, SHAP analysis, and DCA.
3. `05_Biological_Validation.ipynb`: GO/KEGG enrichment and ESTIMATE/TIMER validation.
4. `08_Batch_Effect_and_Subtype_Analysis.ipynb`: PVCA and subtype-adjusted survival.

All code is documented and designed for one-click reproduction of study findings.

**Note:** During the June 2026 restructuring, the original raw GSE matrix files and some intermediate processed CSVs were moved. If data paths are missing, please restore the GSE files into `/Data/raw` from your original backup.

## 🛠️ Usage
### 1. Installation
Install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Running the Analysis
Run the Jupyter notebooks in order (`00` through `07`) to reproduce the study. Note: Notebooks have been updated to reflect the new `/Data/raw` directory structure.

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

## 📜 Publication
The final manuscript is available in `/Manuscript/Manuscript_Final.md`. All methodology follows standard bioinformatics best practices and TRIPOD-AI reporting guidelines for *Computers in Biology and Medicine*.
