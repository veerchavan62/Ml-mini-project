# Online News Popularity: End-to-End Interpretable Machine Learning Framework

An interpretable machine learning framework designed to predict and analyze article virality (social shares) using the UCI Online News Popularity dataset.

---

## 📌 Project Overview
This project provides an end-to-end predictive pipeline that combines:
1. **Regression Modeling**: Estimating expected article shares using log-transformed targets.
2. **Classification & Threshold Optimization**: Identifying high-impact articles ($\ge 1,400$ shares) with cost-sensitive threshold tuning.
3. **Model Interpretability & Feature Attribution**: Combining linear coefficient weights and tree-based gain metrics to understand the key drivers of social sharing.

---

## 📂 Repository Structure

```text
├── figures/                     # Generated visual plots & evaluation curves
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── residuals_and_predictions.png
│   ├── roc_pr_curves.png
│   └── threshold_optimization.png
├── results/                     # Metric evaluation tables & summaries (CSV)
│   ├── category_importance_summary.csv
│   ├── classification_metrics.csv
│   ├── feature_importance_summary.csv
│   └── regression_metrics.csv
├── src/                         # Modular codebase
│   ├── data_pipeline.py         # Data loading, cleaning, preprocessing, and scaling
│   ├── evaluation.py           # Regression & classification evaluation metrics
│   ├── interpretability.py     # Attribution analysis and plot generation
│   └── models.py               # Ridge, ElasticNet, Random Forest, XGBoost
├── OnlineNewsPopularity.csv     # Dataset
├── run_pipeline.py              # Master pipeline execution script
├── show_results.py              # CLI summary report viewer
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/veerchavan62/Ml-mini-project.git
cd Ml-mini-project
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
Execute the end-to-end pipeline:
```bash
python run_pipeline.py
```
This will:
- Clean and preprocess the dataset
- Split into stratified train (80%) and test (20%) sets
- Train regression and classification models
- Perform threshold optimization
- Generate figures in `figures/` and metrics in `results/`

### 3. View Summary Results
To inspect the metrics in the terminal:
```bash
python show_results.py
```

---

## 📊 Key Results

### Regression Models (Expected Shares Estimation)
| Model | RMSLE (RMSE on log) | MAE (log) | $R^2$ Score | MAE (shares) | MedAE (shares) |
|---|---|---|---|---|---|
| **XGBoost Regressor** | **0.8508** | **0.6230** | **0.1770** | **2,503.9** | **697.6** |
| Random Forest | 0.8601 | 0.6323 | 0.1590 | 2,518.7 | 719.4 |
| ElasticNet | 0.8782 | 0.6466 | 0.1232 | 2,540.7 | 727.0 |
| Ridge Regression | 0.8787 | 0.6457 | 0.1222 | 2,540.8 | 725.0 |

### Classification Performance (High-Impact Article Assessment)
| Model / Setup | Accuracy | Bal. Acc | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **Direct XGBoost Classifier** | **67.22%** | **66.87%** | **68.27%** | 72.06% | **0.7012** | **0.7338** | **0.7561** |
| XGBoost Regressor (Default Threshold) | 66.29% | 65.09% | 64.28% | 82.86% | 0.7240 | 0.7243 | 0.7329 |
| XGBoost Regressor (Optimized $\tau^*$) | 65.91% | 64.64% | 63.79% | 83.53% | 0.7234 | 0.7243 | 0.7329 |
| Direct Logistic Regression | 65.85% | 65.52% | 67.20% | 70.31% | 0.6872 | 0.7087 | 0.7250 |
| ElasticNet (Default Threshold) | 62.88% | 61.21% | 60.73% | **86.13%** | 0.7123 | 0.6993 | 0.7057 |

---

## 📈 Visualizations
All generated visualizations are saved in the `figures/` directory:
- `feature_importance.png`: Top influential features by gain and linear weight
- `roc_pr_curves.png`: ROC and Precision-Recall curves
- `confusion_matrices.png`: Confusion matrices across models
- `residuals_and_predictions.png`: Residual plots and calibration
- `threshold_optimization.png`: Classification threshold sweep

---

## 📄 License
This project is for educational and research purposes.

