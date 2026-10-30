# A Compact 4-Gene Signature (FHL1, PCK1, FABP4, MT1H) Defines an Immune-Cold Microenvironment in Breast Cancer

This repository contains the complete, strictly validated, and leak-free computational pipeline for the manuscript: 
**"A Compact 4-Gene Signature (FHL1, PCK1, FABP4, and MT1H) Defines an Immune-Cold Microenvironment and Predicts Diagnostic Outcomes in Breast Cancer"**

## 📌 Project Overview
Breast cancer remains a significant clinical challenge due to its molecular heterogeneity. While multi-gene signatures provide prognostic guidance, many suffer from data leakage and artificially inflated performance metrics. 

In this study, we developed a robust, 4-gene diagnostic signature using a strictly leak-free machine learning pipeline. Crucially, we discovered that this signature identifies an **immune-cold** (CD8+ T-cell excluded) tumor microenvironment, challenging the conventional "immune-hot" narrative. 

## 📊 Key Real Results (No Fabrication, No Leakage)
*   **Validation AUC:** `0.8853`
*   **Balanced Accuracy:** `0.7907`
*   **MCC:** `0.5839`
*   **Brier Score (Calibration):** `0.1414`
*   **FHL1 Expression:** **UP** in tumors (Coefficient: `+0.3502`)
*   **PCK1, FABP4, MT1H Expression:** DOWN in tumors
*   **Immune Microenvironment:** **Immune-Cold** (Spearman correlation with CD8A: `-0.4387`, p = 0.0033)
*   **PVCA Variance:** Technical Batch = `0.02%`, Biological Status = `8.25%`

## 📂 Repository Contents
*   `4Gene_Signature_Pipeline.ipynb` — The complete Jupyter Notebook containing:
    *   Data loading and GPL annotation parsing.
    *   IQR-based probe collapsing (13,101 common genes).
    *   Strict leak-free normalization (fit on Training only).
    *   Differential Expression Analysis (1,542 DEGs) and KEGG Enrichment (PPAR pathway).
    *   Logistic Regression Model (4-gene) and Platt Scaling.
    *   Tumor-only Immune Deconvolution (CD8A, PRF1, etc.).
    *   SHAP Analysis and PVCA.
    *   Generation of all 8 Figures and 3 Tables.
*   `requirements.txt` — The Python dependencies required to run the notebook.
*   `Data/processed/` — Output directory (generated when you run the notebook) containing all high-resolution `.png` figures and `.xlsx` tables.

## 📥 Data Sources
The raw transcriptomic data can be downloaded directly from the NCBI Gene Expression Omnibus (GEO):
*   **Training Cohort:** [GSE42568](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42568) (GPL570)
*   **Validation Cohort:** [GSE15852](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE15852) (GPL96)

*Note: To run the notebook, place the raw `.txt` files inside a folder named `Data/raw/` before executing the cells.*

## ⚙️ How to Reproduce
1. Clone this repository:
   ```bash
   git clone https://github.com/huzaifaakmal/ml-breast-cancer-biomarker.git