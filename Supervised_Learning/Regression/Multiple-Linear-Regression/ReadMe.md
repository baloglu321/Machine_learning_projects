# Multiple Linear Regression

Multiple Linear Regression implementation for predicting continuous outcomes using multiple predictor variables with Streamlit interface.

## 📊 Algorithm Overview

**Multiple Linear Regression** extends simple linear regression to model the relationship between multiple independent variables and a dependent variable: Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ.

### Key Characteristics:
- **Multiple Predictors**: Uses several independent variables
- **Linear Combination**: Weighted sum of features
- **Least Squares**: Minimizes squared residuals
- **Interpretable Coefficients**: Each β shows feature impact
- **Multicollinearity Handling**: Can identify correlated features
- **Versatile**: Applicable to many domains

## ✨ Features

- Interactive Streamlit interface
- Multiple feature inputs
- Coefficient analysis
- R² and adjusted R² metrics
- Residual analysis
- Feature importance visualization

## 🛠️ Technologies

- Python 3.8+
- Streamlit
- scikit-learn - LinearRegression
- pandas, numpy
- matplotlib, seaborn

## 📦 Dataset

Regression dataset with multiple predictor variables.

## 🚀 Installation

```bash
cd Supervised_Learning/Regression/Multiple-Linear-Regression
pip install -r requirements.txt
```

## 💻 Usage

```bash
streamlit run main.py
```

## 📁 Project Structure

```
Multiple-Linear-Regression/
├── main.py          # Streamlit app
├── utils.py         # Model implementation
├── data/            # Dataset
└── models/          # Trained models
```

## 📈 Model Performance

- **R² Score**: Variance explained
- **Adjusted R²**: Penalizes unnecessary features
- **MSE/RMSE**: Prediction error metrics
- **Coefficient Analysis**: Feature importance

## 🎯 Use Cases

- Sales forecasting with multiple factors
- Real estate price prediction
- Energy consumption modeling
- Student performance prediction
- Economic forecasting

## 📝 Formula

```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε
```

---

**Developed by MEB**
