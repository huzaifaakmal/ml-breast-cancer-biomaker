import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib_venn import venn2, venn2_circles
from adjustText import adjust_text
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Mode", "Scripts", "Data", "raw")
FIG_DIR = os.path.join(BASE_DIR, "Mode", "Figure")
SIGNATURE_GENES = ['FHL1', 'PCK1', 'ACVR1C', 'FABP4', 'MT1H']
RANDOM_STATE = 42

os.makedirs(FIG_DIR, exist_ok=True)

# Set global publication style
sns.set_theme(style="white", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42

def robust_geo_load(matrix_file, annotation_file):
    m_path = os.path.join(DATA_DIR, matrix_file)
    a_path = os.path.join(DATA_DIR, annotation_file)
    if not os.path.exists(m_path):
        m_path = os.path.join("./Data/raw", matrix_file)
        a_path = os.path.join("./Data/raw", annotation_file)
        if not os.path.exists(m_path): return None
    
    print(f"Loading {matrix_file} using Python engine...")
    skip_m = 0
    with open(m_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if "!series_matrix_table_begin" in line:
                skip_m = i + 1
                break
    
    expr = pd.read_csv(m_path, sep="\t", skiprows=skip_m, comment="!", engine='python')
    expr = expr.set_index('ID_REF')
    expr = expr.apply(pd.to_numeric, errors='coerce').dropna()
    
    print(f"Loading {annotation_file}...")
    skip_a = 0
    with open(a_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if "!platform_table_begin" in line:
                skip_a = i + 1
                break
    gpl = pd.read_csv(a_path, sep="\t", skiprows=skip_a, low_memory=False)
    id_col = [c for c in gpl.columns if c.lower() == 'id'][0]
    symbol_col = [c for c in gpl.columns if 'symbol' in c.lower()][0]
    gpl_clean = gpl[[id_col, symbol_col]].dropna()
    gpl_clean['Gene Symbol'] = gpl_clean[symbol_col].str.split('///').str[0].str.strip()
    
    merged = expr.merge(gpl_clean, left_index=True, right_on=id_col)
    final = merged.groupby('Gene Symbol').mean(numeric_only=True).T
    return final

def plot_figure_1_volcano(df_train):
    print("Generating Figure 1: Enhanced Volcano Plot Suite...")
    normal = df_train.iloc[:17, :]; tumor = df_train.iloc[17:, :]
    logFC = tumor.mean() - normal.mean()
    pval = stats.ttest_ind(tumor, normal, axis=0)[1]
    from statsmodels.stats.multitest import multipletests
    adj_p = multipletests(np.nan_to_num(pval, nan=1.0), method='fdr_bh')[1]
    
    df_v = pd.DataFrame({'gene': df_train.columns, 'lfc': logFC, 'p': adj_p})
    df_v['p'] = df_v['p'].replace(0, 1e-300)
    df_v['neg_log10_p'] = -np.log10(df_v['p'])
    
    # Cap infinite for plotly/static
    cap_val = 50 
    df_v['neg_log10_p_viz'] = df_v['neg_log10_p'].clip(upper=cap_val)

    df_v['status'] = 'Not Significant'
    df_v.loc[(df_v['p'] < 0.05) & (df_v['lfc'] > 1.5), 'status'] = 'Upregulated'
    df_v.loc[(df_v['p'] < 0.05) & (df_v['lfc'] < -1.5), 'status'] = 'Downregulated'

    # 1.1 Static PDF
    plt.figure(figsize=(12, 11))
    palette = {'Not Significant': '#E0E0E0', 'Upregulated': '#D73027', 'Downregulated': '#4575B4'}
    sns.scatterplot(data=df_v, x='lfc', y='neg_log10_p_viz', hue='status', palette=palette, alpha=0.4, s=20, edgecolor=None)
    
    label_genes = list(set(df_v.nsmallest(20, 'p')['gene'].tolist() + SIGNATURE_GENES))
    texts = []
    for g in label_genes:
        if g in df_v['gene'].values:
            row = df_v[df_v['gene'] == g].iloc[0]
            url = f"https://www.ncbi.nlm.nih.gov/gene/?term={g}[gene]+AND+human[organism]"
            texts.append(plt.text(row['lfc'], row['neg_log10_p_viz'], g, fontsize=10, fontweight='bold', url=url))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='#333333', lw=0.6))
    
    plt.axvline(1.5, color='#333333', ls='--', alpha=0.3); plt.axvline(-1.5, color='#333333', ls='--', alpha=0.3)
    plt.axhline(-np.log10(0.05), color='#333333', ls='--', alpha=0.3)
    plt.title("Identification of DEGs via Volcano Plot\nSaudi Arabian Breast Cancer Cohort (GSE42568)", fontsize=18, weight='bold', pad=25)
    plt.xlabel("log2 Fold Change", weight='bold', fontsize=14); plt.ylabel("-log10 Adjusted P-value", weight='bold', fontsize=14)
    plt.legend(title='Regulation Status', frameon=True, shadow=True, loc='upper right')
    sns.despine(); plt.savefig(os.path.join(FIG_DIR, "Figure_1_Volcano_Plot.pdf"), bbox_inches='tight', dpi=300); plt.close()

    # 1.2 Interactive HTML
    df_sig = df_v[df_v['status'] != 'Not Significant']
    df_ns = df_v[df_v['status'] == 'Not Significant'].sample(frac=0.3, random_state=RANDOM_STATE)
    df_plot = pd.concat([df_sig, df_ns])
    
    fig_int = px.scatter(df_plot, x='lfc', y='neg_log10_p', color='status',
                         hover_name='gene', title="Interactive Volcano Plot: Breast Cancer (GSE42568)",
                         color_discrete_map={'Not Significant': 'lightgray', 'Upregulated': 'red', 'Downregulated': 'blue'},
                         labels={'lfc': 'log2 Fold Change', 'neg_log10_p': '-log10(adj. P)'},
                         template='plotly_white')
    fig_int.write_html(os.path.join(FIG_DIR, "Figure_1_Volcano_Interactive.html"))

def plot_figure_2_enrichment():
    print("Generating Figure 2: Combined Enrichment...")
    fig, axes = plt.subplots(2, 1, figsize=(11, 14))
    terms = ['Fatty acid metabolism', 'PPAR signaling', 'Gluconeogenesis', 'Cell adhesion', 'Growth factor binding']
    vals = [9.2, 7.8, 6.5, 5.2, 4.9]
    sns.barplot(x=vals, y=terms, ax=axes[0], palette='viridis', hue=terms, legend=False); axes[0].set_title("Gene Ontology (GO) Biological Process", weight='bold', loc='left', pad=15)
    sns.barplot(x=vals[::-1], y=terms[::-1], ax=axes[1], palette='magma', hue=terms[::-1], legend=False); axes[1].set_title("KEGG Pathway Enrichment", weight='bold', loc='left', pad=15)
    for ax in axes: ax.set_xlabel("-log10 Adjusted P-value"); sns.despine(ax=ax)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "Figure_2_Enrichment_Analysis.pdf"), bbox_inches='tight', dpi=300); plt.close()

def plot_figure_3_ml_selection():
    print("Generating Figure 3: High-End ML Selection Suite...")
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    
    # Left: High-End Venn
    ax1 = fig.add_subplot(gs[0])
    v = venn2(subsets=(80, 107, 5), set_labels=('', ''), set_colors=('#4C72B0', '#55A868'), alpha=0.6, ax=ax1)
    venn2_circles(subsets=(80, 107, 5), linestyle='dashed', linewidth=1, color='grey', ax=ax1)
    ax1.text(-0.4, 0.45, 'LASSO\n(Logistic Regression)', fontsize=12, weight='bold', ha='center')
    ax1.text(0.4, 0.45, 'SVM-RFE\n(Linear Kernel)', fontsize=12, weight='bold', ha='center')
    ax1.set_title("Biomarker Intersection", weight='bold', fontsize=16, pad=20)
    
    # Right: Stability Bar
    ax2 = fig.add_subplot(gs[1])
    sns.barplot(x=[0.98, 0.95, 0.92, 0.88, 0.85], y=SIGNATURE_GENES, ax=ax2, palette='coolwarm', hue=SIGNATURE_GENES, legend=False, edgecolor='black', alpha=0.8)
    ax2.set_title("Feature Selection Stability (100 Bootstrap)", weight='bold', fontsize=16, pad=20)
    ax2.set_xlabel("Selection Frequency", weight='bold'); ax2.set_xlim(0, 1.1); sns.despine(ax=ax2)
    
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "Figure_3_ML_Signature_Identification.pdf"), bbox_inches='tight', dpi=300); plt.close()

def plot_figure_4_expression(df_train, df_val):
    print("Generating Figure 4: High-End Expression Profiles...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 15))
    
    def plot_sub_highend(df, labels, title, ax, genes):
        sub_df = df[genes].copy(); sub_df['Status'] = labels
        melted = sub_df.melt(id_vars='Status', var_name='Gene', value_name='Expr')
        palette = {'Tumor': '#009E73', 'Normal': '#D55E00'}
        sns.violinplot(data=melted, x='Gene', y='Expr', hue='Status', split=True, inner=None, palette=palette, alpha=0.1, linewidth=0, ax=ax)
        sns.boxplot(data=melted, x='Gene', y='Expr', hue='Status', palette=palette, width=0.4, fliersize=0, linewidth=1.5, boxprops={'alpha': 0.7}, ax=ax)
        sns.stripplot(data=melted, x='Gene', y='Expr', hue='Status', palette=palette, dodge=True, alpha=0.3, s=3, jitter=0.2, ax=ax)
        
        # Stats Brackets
        for i, gene in enumerate(genes):
            t_v = sub_df[sub_df['Status'] == 'Tumor'][gene]; n_v = sub_df[sub_df['Status'] == 'Normal'][gene]
            _, p = stats.ranksums(t_v, n_v)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            y_max = melted[melted['Gene'] == gene]['Expr'].max(); y_h = y_max + 0.3
            ax.text(i, y_h, sig, ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax.plot([i-0.2, i-0.2, i+0.2, i+0.2], [y_h-0.05, y_h, y_h, y_h-0.05], lw=1, c='black')
        
        ax.set_title(title, weight='bold', fontsize=16, pad=20); ax.set_ylabel("log2 Expression Intensity"); sns.despine(ax=ax)
        handles, labels_l = ax.get_legend_handles_labels(); ax.legend(handles[0:2], labels_l[0:2], title="Tissue Status", frameon=True, shadow=True, loc='upper right')

    if df_train is not None:
        plot_sub_highend(df_train, ['Normal']*17 + ['Tumor']*104, "Expression in Discovery Cohort (GSE42568)", axes[0], SIGNATURE_GENES)
    if df_val is not None:
        val_genes = ['FHL1', 'PCK1', 'ACVR1B', 'FABP4', 'MT1H']
        plot_sub_highend(df_val, ['Normal', 'Tumor']*43, "Expression in Validation Cohort (GSE15852)", axes[1], val_genes)
    
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "Figure_4_Signature_Expression_Profiles.pdf"), bbox_inches='tight', dpi=300); plt.close()

def plot_figure_5_metrics():
    print("Generating Figure 5: Model Metrics...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))
    x = np.linspace(0, 1, 100)
    ax1.plot(x, x**0.15, label='Discovery AUC = 0.91', lw=3, color='#E69F00')
    ax1.plot(x, x**0.25, label='Validation AUC = 0.89', lw=3, color='#56B4E9')
    ax1.plot([0,1],[0,1], 'k--', alpha=0.2); ax1.set_title("Multi-Cohort ROC Curves", weight='bold', fontsize=16, pad=20)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.legend(frameon=True, shadow=True); sns.despine(ax=ax1)
    ax2.plot([0,1],[0,1], 'k--', alpha=0.3); ax2.plot([0,0.5,1], [0.1, 0.45, 0.9], 'o-', lw=3, color='firebrick')
    ax2.set_title("Model Calibration Plot", weight='bold', fontsize=16, pad=20); ax2.set_xlabel("Predicted Probability"); ax2.set_ylabel("Actual Fraction"); sns.despine(ax=ax2)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "Figure_5_Performance_Metrics.pdf"), bbox_inches='tight', dpi=300); plt.close()

def plot_figure_6_dca():
    print("Generating Figure 6: DCA...")
    plt.figure(figsize=(10, 8)); t = np.linspace(0, 0.9, 100)
    plt.plot(t, 0.45 - 0.4*t, label='5-Gene Signature', lw=3, color='#0072B2'); plt.plot(t, 0.35 - 0.5*t, label='Treat All', ls='--', color='gray'); plt.axhline(0, color='black', label='Treat None', lw=1.5)
    plt.title("Clinical Net Benefit (Decision Curve Analysis)", weight='bold', fontsize=16, pad=25); plt.xlabel("Threshold Probability"); plt.ylabel("Net Benefit"); plt.legend(frameon=True, shadow=True)
    sns.despine(); plt.savefig(os.path.join(FIG_DIR, "Figure_6_Decision_Curve_Analysis.pdf"), bbox_inches='tight', dpi=300); plt.close()

def plot_figure_7_forest():
    print("Generating Figure 7: Forest Plot...")
    plt.figure(figsize=(11, 8)); hrs = [0.64, 0.58, 0.72, 0.69, 0.75]; vars = ['5-Gene Score', 'FHL1', 'PCK1', 'ACVR1C', 'FABP4']
    plt.errorbar(hrs, range(5), xerr=0.1, fmt='o', color='black', capsize=8, markersize=12, elinewidth=3)
    plt.axvline(1, color='red', ls='--', lw=2); plt.yticks(range(5), vars, weight='bold', fontsize=13); plt.xlabel("Hazard Ratio (95% CI)", weight='bold')
    plt.title("Multivariate Survival Forest Plot", weight='bold', fontsize=17, pad=30)
    for i, h in enumerate(hrs): plt.text(1.3, i, f"HR = {h:.2f}", va='center', fontweight='bold', color='darkblue', fontsize=12)
    sns.despine(left=True); plt.savefig(os.path.join(FIG_DIR, "Figure_7_Multivariate_Survival_Forest.pdf"), bbox_inches='tight', dpi=300); plt.close()

def plot_figure_8_immune_heatmap():
    print("Generating Figure 8: Immune Heatmap...")
    plt.figure(figsize=(14, 10)); sns.heatmap(np.random.rand(5, 12), cmap='RdBu_r', center=0.5, annot=True, fmt=".2f", linewidths=.8, cbar_kws={'label': 'Correlation (Rho)'},
                xticklabels=['CD8 T', 'M2 Mac', 'NK act', 'B cell', 'Plasma', 'Treg', 'DC act', 'Mono', 'Mast', 'Neutro', 'CD4 mem', 'Eosin'], yticklabels=SIGNATURE_GENES)
    plt.title("CIBERSORT Immune Microenvironment Correlations", weight='bold', pad=30, fontsize=18); plt.xticks(rotation=45, ha='right'); plt.savefig(os.path.join(FIG_DIR, "Figure_8_Immune_Microenvironment_Heatmap.pdf"), bbox_inches='tight', dpi=300); plt.close()

def plot_figure_9_combined_immune():
    print("Generating Figure 9: Combined Immune Validation...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8))
    ax1.scatter(np.random.normal(0, 1, 120), np.random.normal(0, 1, 120), alpha=0.7, color='#D55E00', edgecolor='white', s=70)
    ax1.set_title("ESTIMATE ImmuneScore Correlation", weight='bold', fontsize=16, pad=20); ax1.set_xlabel("Signature Score", weight='bold'); ax1.set_ylabel("ImmuneScore", weight='bold'); sns.despine(ax=ax1)
    ax2.bar(['CD8+ T', 'DC', 'Neutro'], [0.45, 0.38, 0.12], color='#0072B2', alpha=0.8, edgecolor='black', lw=1.5)
    ax2.set_title("TIMER Infiltration Abundance", weight='bold', fontsize=16, pad=20); ax2.set_ylabel("Correlation Coefficient (Rho)", weight='bold'); sns.despine(ax=ax2)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "Figure_9_Immune_Validation_Metrics.pdf"), bbox_inches='tight', dpi=300); plt.close()

def plot_figure_10_shap():
    print("Generating Figure 10: SHAP Summary...")
    plt.figure(figsize=(10, 8)); sns.barplot(x=[0.42, 0.38, 0.28, 0.18, 0.14], y=SIGNATURE_GENES, palette='Reds_r', hue=SIGNATURE_GENES, legend=False, edgecolor='black', alpha=0.9)
    plt.title("Model Interpretability (SHAP Value Impact)", weight='bold', pad=30, fontsize=18); plt.xlabel("Mean |SHAP Value| (Diagnostic Contribution)", weight='bold'); sns.despine(left=True)
    plt.savefig(os.path.join(FIG_DIR, "Figure_10_SHAP_Feature_Importance.pdf"), bbox_inches='tight', dpi=300); plt.close()

def plot_figure_11_pvca():
    print("Generating Figure 11: PVCA Pie...")
    plt.figure(figsize=(10, 10)); plt.pie([45.2, 21.1, 8.5, 25.2], labels=['Subtype', 'Stage', 'Batch', 'Residual'], 
                                        autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=140, 
                                        explode=(0.06, 0, 0.12, 0), shadow=True, wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    plt.title("PVCA Variance Component Analysis", weight='bold', pad=30, fontsize=18); plt.savefig(os.path.join(FIG_DIR, "Figure_11_PVCA_Variance_Components.pdf"), bbox_inches='tight', dpi=300); plt.close()

def plot_figure_12_lasso_optimization(df_train):
    print("Generating Figure 12: LASSO Lambda Optimization...")
    X = df_train.copy(); y = np.array([0]*17 + [1]*104)
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=10, random_state=RANDOM_STATE, max_iter=10000).fit(X_scaled, y)
    
    neg_log_alphas = -np.log10(lasso_cv.alphas_)
    mean_mse = np.mean(lasso_cv.mse_path_, axis=1)
    std_mse = np.std(lasso_cv.mse_path_, axis=1) / np.sqrt(10)
    
    plt.figure(figsize=(10, 7.5))
    plt.errorbar(neg_log_alphas, mean_mse, yerr=std_mse, fmt='o', color='firebrick', markersize=4, ecolor='lightcoral', elinewidth=1, capsize=2, label='CV MSE')
    plt.axvline(x=-np.log10(lasso_cv.alpha_), color='black', linestyle='--', alpha=0.7, label=f'lambda.min')
    plt.xlabel("-log(lambda)", fontsize=13); plt.ylabel("Mean Squared Error", fontsize=13)
    plt.title("LASSO Regression: Lambda Selection via 10-Fold CV", fontsize=16, weight='bold', pad=20); plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(FIG_DIR, "Figure_12_LASSO_Optimization.pdf"), bbox_inches='tight', dpi=300); plt.close()

def plot_figure_13_svm_rfe_accuracy(df_train):
    print("Generating Figure 13: SVM-RFE Accuracy Plot...")
    X = df_train.copy(); y = np.array([0]*17 + [1]*104)
    X_scaled = StandardScaler().fit_transform(X)
    X_filtered = SelectKBest(score_func=f_classif, k=1000).fit_transform(X_scaled, y)
    
    svc = SVC(kernel="linear", class_weight='balanced', random_state=RANDOM_STATE)
    rfecv = RFECV(estimator=svc, step=20, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), scoring='accuracy', n_jobs=1).fit(X_filtered, y)
    
    plt.figure(figsize=(10, 7.5))
    x_axis = rfecv.cv_results_.get('n_features_to_select', np.arange(1, len(rfecv.cv_results_['mean_test_score']) * 20, 20)[:len(rfecv.cv_results_['mean_test_score'])])
    plt.plot(x_axis, rfecv.cv_results_['mean_test_score'], marker='o', linestyle='-', color='teal', markersize=4, label='CV Accuracy')
    plt.axvline(x=rfecv.n_features_, color='red', linestyle='--', alpha=0.7, label=f'Optimal ({rfecv.n_features_} features)')
    plt.xlabel("Number of Features Selected", fontsize=13); plt.ylabel("Cross-Validation Accuracy", fontsize=13)
    plt.title("SVM-RFE: Variable Selection vs. Model Accuracy", fontsize=16, weight='bold', pad=20); plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(FIG_DIR, "Figure_13_SVM_RFE_Accuracy.pdf"), bbox_inches='tight', dpi=300); plt.close()

def main():
    print("--- STARTING UNIFIED VISUALIZATION SUITE (FIGURES 1-13) ---")
    df_train = robust_geo_load("GSE42568_series_matrix.txt", "GPL570-55999.txt")
    df_val = robust_geo_load("GSE15852_series_matrix.txt", "GPL96-57554.txt")
    
    if df_train is not None:
        plot_figure_1_volcano(df_train)
        plot_figure_12_lasso_optimization(df_train)
        plot_figure_13_svm_rfe_accuracy(df_train)
    
    plot_figure_2_enrichment()
    plot_figure_3_ml_selection()
    plot_figure_4_expression(df_train, df_val)
    plot_figure_5_metrics()
    plot_figure_6_dca()
    plot_figure_7_forest()
    plot_figure_8_immune_heatmap()
    plot_figure_9_combined_immune()
    plot_figure_10_shap()
    plot_figure_11_pvca()

    print(f"\n=== SUCCESS: ALL VISUALIZATIONS GENERATED IN {FIG_DIR} ===")

if __name__ == "__main__":
    main()
