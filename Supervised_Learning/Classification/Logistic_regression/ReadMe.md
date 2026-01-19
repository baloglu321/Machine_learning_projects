# Logistic Regression Customer Segmentation

Logistic Regression implementation for customer segmentation with Streamlit. A fundamental statistical method for binary and multiclass classification.

## 📊 Algorithm Overview

**Logistic Regression** is a statistical method for predicting binary or multiclass outcomes. Despite its name, it's a classification algorithm that uses the logistic function to model probability.

### Key Characteristics:
- **Probabilistic**: Outputs probability estimates
- **Linear Decision Boundary**: Separates classes linearly
- **Interpretable**: Coefficients show feature importance
- **Fast**: Quick training and prediction
- **Regularization**: L1 (Lasso) and L2 (Ridge) support
- **Multiclass Support**: One-vs-Rest or Multinomial

## ✨ Features

- Interactive Streamlit interface
- Probability-based classification
- Feature coefficient analysis
- Real-time predictions
- Regularization options
- Model interpretability

## 🛠️ Technologies

- Python 3.8+
- Streamlit
- scikit-learn - LogisticRegression
- pandas, numpy
- matplotlib

## 📦 Dataset

Customer segmentation dataset.

**Data Files:**
- `data/Train.csv`
- `data/Test.csv`

## 🚀 Installation

```bash
cd Supervised_Learning/Classification/Logistic_regression
pip install -r requirements.txt
```

## 💻 Usage

```bash
streamlit run Home.py
```

## 📁 Project Structure

```
Logistic_regression/
├── Home.py          # Streamlit application
├── utils.py         # Logistic Regression logic
├── data/            # Datasets
└── models/          # Saved models
```

## 📈 Model Performance

- **Accuracy**: Good for linearly separable data
- **Coefficients**: Feature importance weights
- **Probability Scores**: Confidence in predictions
- **Regularization**: Prevents overfitting

## 🎯 Use Cases

- Customer segmentation
- Email spam detection
- Disease diagnosis
- Credit scoring
- Marketing response prediction

## ⚙️ Hyperparameters

- `penalty`: L1, L2, or ElasticNet regularization
- `C`: Inverse regularization strength
- `solver`: Optimization algorithm
- `multi_class`: OvR or multinomial strategy

---

**Developed by MEB**
