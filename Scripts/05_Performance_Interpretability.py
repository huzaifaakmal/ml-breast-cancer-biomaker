import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import shap

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Mode", "Scripts", "Data", "raw")
FIG_DIR = os.path.join(BASE_DIR, "Mode", "Figure")
MODEL_PATH = os.path.join(BASE_DIR, "models", "breast_cancer_pipeline_v1.pkl")
RANDOM_STATE = 42

os.makedirs(FIG_DIR, exist_ok=True)

def process_geo_data(matrix_file, annotation_file, data_dir=DATA_DIR):
    # (Same as script 03)
    m_path = os.path.join(data_dir, matrix_file)
    a_path = os.path.join(data_dir, annotation_file)
    skip = 0
    with open(m_path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith("!series_matrix_table_begin"):
                skip = i + 1
                break
    expr = pd.read_csv(m_path, sep="\t", skiprows=skip, comment="!", low_memory=False)
    gpl = pd.read_csv(a_path, sep="\t", comment="#", low_memory=False)
    gpl_clean = gpl[["ID", "Gene Symbol"]].dropna()
    merged = expr.merge(gpl_clean, left_on="ID_REF", right_on="ID")
    merged["Gene Symbol"] = merged["Gene Symbol"].astype(str).str.split("///").str[0].str.strip()
    return merged.groupby("Gene Symbol").mean(numeric_only=True).T

def calculate_net_benefit(y_true, y_probs, thresholds):
    net_benefit = []
    n = len(y_true)
    for pt in thresholds:
        tp = np.sum((y_probs >= pt) & (y_true == 1))
        fp = np.sum((y_probs >= pt) & (y_true == 0))
        if pt == 1:
            nb = 0
        else:
            nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefit.append(nb)
    return net_benefit

def main():
    # 1. Load Model and Data
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run 03_Model_Pipeline.py first.")
        return

    pipe = joblib.load(MODEL_PATH)
    X_train_raw = process_geo_data("GSE42568_series_matrix.txt", "GPL570-55999.txt")
    y_train = np.array([1]*104 + [0]*17)
    
    X_test_raw = process_geo_data("GSE15852_series_matrix.txt", "GPL96-57554.txt")
    n_test = len(X_test_raw)
    y_test = np.array([1]*(n_test//2) + [0]*(n_test - n_test//2))
    
    common = X_train_raw.columns.intersection(X_test_raw.columns)
    X_test = X_test_raw[common]

    # 2. Figure 6: ROC Curves
    plt.figure(figsize=(8, 8))
    y_probs_train = pipe.predict_proba(X_train_raw[common])[:, 1]
    fpr_t, tpr_t, _ = roc_curve(y_train, y_probs_train)
    plt.plot(fpr_t, tpr_t, label=f'Discovery (AUC = {auc(fpr_t, tpr_t):.2f})')
    
    y_probs_test = pipe.predict_proba(X_test)[:, 1]
    fpr_v, tpr_v, _ = roc_curve(y_test, y_probs_test)
    plt.plot(fpr_v, tpr_v, label=f'Validation (AUC = {auc(fpr_v, tpr_v):.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Figure 6: Diagnostic ROC Curves')
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "fig6.pdf"), bbox_inches='tight')
    plt.close()

    # 3. Figure 8: Decision Curve Analysis (DCA)
    thresholds = np.linspace(0, 0.99, 100)
    nb_model = calculate_net_benefit(y_test, y_probs_test, thresholds)
    nb_all = calculate_net_benefit(y_test, np.ones(len(y_test)), thresholds)
    nb_none = np.zeros(len(thresholds))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, nb_model, label='5-Gene Signature', color='blue', lw=2)
    plt.plot(thresholds, nb_all, label='Treat All', color='gray', linestyle='--')
    plt.plot(thresholds, nb_none, label='Treat None', color='black', lw=1)
    plt.ylim(-0.05, 0.6)
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Figure 8: Decision Curve Analysis (DCA)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, "fig8.pdf"), bbox_inches='tight')
    plt.close()

    # 4. Figure 12: SHAP Summary Plot
    # Pre-transform data for SHAP
    X_test_scaled = pipe.named_steps['scaler'].transform(X_test)
    X_test_sel = pipe.named_steps['selector'].transform(X_test_scaled)
    mask = pipe.named_steps['selector'].get_support()
    sel_features = X_test.columns[mask]
    
    explainer = shap.LinearExplainer(pipe.named_steps['classifier'], X_test_sel)
    shap_values = explainer.shap_values(X_test_sel)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_sel, feature_names=sel_features, show=False)
    plt.title("Figure 12: SHAP Feature Contribution (Linear SVM)")
    plt.savefig(os.path.join(FIG_DIR, "fig12.pdf"), bbox_inches='tight')
    plt.close()
    
    print("Figures 6, 8, and 12 generated.")

if __name__ == "__main__":
    main()
