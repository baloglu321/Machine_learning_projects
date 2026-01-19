# Lasso Regression

Lasso (Least Absolute Shrinkage and Selection Operator) Regression for feature selection and regularized prediction with Streamlit interface.

## 📊 Algorithm Overview

**Lasso Regression** performs L1 regularization, adding a penalty equal to the absolute value of coefficients. This can shrink some coefficients to exactly zero, effectively performing feature selection.

### Key Characteristics:
- **L1 Regularization**: |β| penalty term
- **Feature Selection**: Can eliminate features (β = 0)
- **Sparse Models**: Produces simple, interpretable models
- **Prevents Overfitting**: Regularization reduces variance
- **Handles Multicollinearity**: Selects one from correlated features
- **Automatic Feature Selection**: No manual feature engineering

## ✨ Features

- Interactive Streamlit interface
- Automatic feature selection
- L1 regularization
- Coefficient shrinkage visualization
- Model sparsity analysis
- Cross-validation for α selection

## 🛠️ Technologies

- Python 3.8+
- Streamlit
- scikit-learn - Lasso
- pandas, numpy
- matplotlib

## 📦 Dataset

Regression dataset with potential feature redundancy.

## 🚀 Installation

```bash
cd Supervised_Learning/Regression/Lasso_regression
pip install -r requirements.txt
```

## 💻 Usage

```bash
streamlit run main.py
```

## 📁 Project Structure

```
Lasso_regression/
├── main.py          # Streamlit application
├── utils.py         # Lasso implementation
├── data/            # Dataset
└── models/          # Trained models
```

## 📈 Model Performance

- **R² Score**: Model fit quality
- **Feature Sparsity**: Number of non-zero coefficients
- **Alpha (α)**: Regularization strength
- **Cross-Validation**: Optimal α selection

## 🎯 Use Cases

- High-dimensional data with feature selection needs
- Genomics and bioinformatics
- Text classification with sparse features
- Economic modeling
- Any regression task requiring interpretability

## ⚙️ Hyperparameters

- `alpha (α)`: Regularization strength (higher = more regularization)
- `max_iter`: Maximum iterations for convergence

## 📝 Formula

```
Loss = MSE + α × Σ|βᵢ|
```

Where α controls regularization strength.

---

**Developed by MEB**
