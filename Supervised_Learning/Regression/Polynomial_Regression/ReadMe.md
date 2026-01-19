# Polynomial Regression

Polynomial Regression for modeling non-linear relationships using polynomial features with Streamlit interface.

## 📊 Algorithm Overview

**Polynomial Regression** fits a polynomial equation to data by transforming features into polynomial terms. It can model non-linear relationships while still using linear regression techniques.

### Key Characteristics:
- **Non-Linear Modeling**: Captures curved relationships
- **Feature Transformation**: Creates polynomial features (x², x³, etc.)
- **Flexible**: Degree controls model complexity
- **Linear in Parameters**: Still uses linear regression
- **Risk of Overfitting**: High degrees can overfit
- **Interpretable**: When using low degrees

## ✨ Features

- Interactive Streamlit interface
- Non-linear relationship modeling
- Adjustable polynomial degree
- Feature transformation pipeline
- Overfitting detection
- Visual curve fitting

## 🛠️ Technologies

- Python 3.8+
- Streamlit
- scikit-learn - PolynomialFeatures, LinearRegression
- pandas, numpy
- matplotlib

## 📦 Dataset

**Car Price Prediction Dataset**

File: `data/car_price_prediction.csv`

Regression task with non-linear feature relationships.

## 🚀 Installation

```bash
cd Supervised_Learning/Regression/Polynomial_Regression
pip install -r requirements.txt
```

## 💻 Usage

```bash
streamlit run Home.py
```

## 📁 Project Structure

```
Polynomial_Regression/
├── Home.py          # Streamlit application
├── utils.py         # Polynomial regression logic
├── data/
│   └── car_price_prediction.csv
└── models/          # Trained models
```

## 📈 Model Performance

- **R² Score**: Model fit quality
- **Degree Selection**: Balance complexity and overfitting
- **MSE**: Prediction error
- **Visual Fit**: Curve alignment with data

## 🎯 Use Cases

- Car price prediction
- Growth curve modeling
- Physical phenomena (projectile motion, etc.)
- Economic trends
- Any relationship with curvature

## ⚙️ Hyperparameters

- `degree`: Polynomial degree (2-5 typically, higher = more complex)
- `include_bias`: Include intercept term

## 📝 Formula

For degree = 2:
```
Y = β₀ + β₁X + β₂X² + ε
```

For degree = 3:
```
Y = β₀ + β₁X + β₂X² + β₃X³ + ε
```

## ⚠️ Caution

- **High Degrees**: Can cause severe overfitting
- **Extrapolation**: Predictions outside training range can be unreliable
- **Scaling**: Feature scaling recommended

---

**Developed by MEB**
