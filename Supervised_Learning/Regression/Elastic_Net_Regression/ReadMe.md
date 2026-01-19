# Elastic Net Regression

Elastic Net Regression combining L1 and L2 regularization for robust feature selection and coefficient shrinkage with Streamlit interface.

## 📊 Algorithm Overview

**Elastic Net** combines the penalties of Ridge (L2) and Lasso (L1) regression. It provides a balance between feature selection and coefficient shrinkage, overcoming limitations of both methods.

### Key Characteristics:
- **Combined Regularization**: L1 + L2 penalties
- **Feature Selection**: Like Lasso
- **Grouped Selection**: Selects correlated features together
- **Stable**: More robust than Lasso alone
- **Flexible**: Tunable L1/L2 ratio
- **Best of Both Worlds**: Feature selection + stability

## ✨ Features

- Interactive Streamlit interface
- Combined L1 and L2 regularization
- Automatic feature selection
- Handles multicollinearity
- Grouped feature selection
- Cross-validation for hyperparameters

## 🛠️ Technologies

- Python 3.8+
- Streamlit
- scikit-learn - ElasticNet
- pandas, numpy
- matplotlib

## 📦 Dataset

Regression dataset with correlated features.

## 🚀 Installation

```bash
cd Supervised_Learning/Regression/Elastic_Net_Regression
pip install -r requirements.txt
```

## 💻 Usage

```bash
streamlit run main.py
```

## 📁 Project Structure

```
Elastic_Net_Regression/
├── main.py          # Streamlit application
├── utils.py         # Elastic Net implementation
├── data/            # Dataset
└── models/          # Trained models
```

## 📈 Model Performance

- **R² Score**: Model fit
- **Feature Sparsity**: Selected features
- **Alpha (α)**: Overall regularization strength
- **L1 Ratio**: Balance between L1 and L2

## 🎯 Use Cases

- High-dimensional data with feature groups
- Genomics with correlated gene expressions
- Financial modeling with correlated indicators
- Marketing with multiple related channels
- Any task requiring both feature selection and stability

## ⚙️ Hyperparameters

- `alpha (α)`: Overall regularization strength
- `l1_ratio`: Mix of L1 and L2 (0 = Ridge, 1 = Lasso, 0.5 = equal mix)

## 📝 Formula

```
Loss = MSE + α × [l1_ratio × Σ|βᵢ| + (1-l1_ratio) × Σβᵢ²]
```

## 🔄 Comparison

| Method | L1 | L2 | Feature Selection | Grouped Selection |
|--------|----|----|-------------------|-------------------|
| Lasso | ✓ | ✗ | ✓ | ✗ |
| Ridge | ✗ | ✓ | ✗ | ✗ |
| Elastic Net | ✓ | ✓ | ✓ | ✓ |

---

**Developed by MEB**
