# Ridge Regression

Ridge Regression implementation with L2 regularization for handling multicollinearity and preventing overfitting with Streamlit interface.

## 📊 Algorithm Overview

**Ridge Regression** performs L2 regularization, adding a penalty equal to the square of coefficients. This shrinks coefficients but doesn't eliminate them, making the model more stable.

### Key Characteristics:
- **L2 Regularization**: β² penalty term
- **Reduces Multicollinearity**: Handles correlated features well
- **Coefficient Shrinkage**: All coefficients shrink but remain non-zero
- **Stable Predictions**: Reduces model variance
- **No Feature Selection**: Keeps all features
- **Closed-Form Solution**: Efficient computation

## ✨ Features

- Interactive Streamlit interface
- L2 regularization
- Multicollinearity handling
- Coefficient stability analysis
- Cross-validation for α selection
- Regularization path visualization

## 🛠️ Technologies

- Python 3.8+
- Streamlit
- scikit-learn - Ridge
- pandas, numpy
- matplotlib

## 📦 Dataset

Regression dataset, particularly effective with correlated features.

## 🚀 Installation

```bash
cd Supervised_Learning/Regression/Ridge-regression
pip install -r requirements.txt
```

## 💻 Usage

```bash
streamlit run main.py
```

## 📁 Project Structure

```
Ridge-regression/
├── main.py          # Streamlit application
├── utils.py         # Ridge implementation
├── data/            # Dataset
└── models/          # Trained models
```

## 📈 Model Performance

- **R² Score**: Model quality
- **Coefficient Shrinkage**: Degree of regularization
- **Alpha (α)**: Regularization parameter
- **Cross-Validation Score**: Generalization performance

## 🎯 Use Cases

- Data with multicollinearity
- Small dataset with many features
- Economic forecasting
- Medical research
- Any regression with correlated predictors

## ⚙️ Hyperparameters

- `alpha (α)`: Regularization strength
- `solver`: Optimization method

## 📝 Formula

```
Loss = MSE + α × Σβᵢ²
```

## 🔄 Ridge vs Lasso

| Feature | Ridge | Lasso |
|---------|-------|-------|
| Penalty | L2 (β²) | L1 (β) |
| Feature Selection | No | Yes |
| Coefficient Behavior | Shrinks all | Can zero out |
| Best For | Multicollinearity | Feature selection |

---

**Developed by MEB**
