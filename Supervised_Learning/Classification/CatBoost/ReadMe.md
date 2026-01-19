# CatBoost Customer Segmentation

CatBoost (Categorical Boosting) implementation for customer segmentation. An advanced gradient boosting library specifically designed to handle categorical features efficiently.

## 📊 Algorithm Overview

**CatBoost** is a gradient boosting algorithm developed by Yandex that excels at handling categorical features without extensive preprocessing. It uses ordered boosting and an innovative algorithm for processing categorical features.

### Key Characteristics:
- **Categorical Feature Handling**: Native support without encoding
- **Ordered Boosting**: Reduces overfitting
- **Symmetric Trees**: Faster training and prediction
- **GPU Support**: Accelerated training
- **Robust**: Less prone to overfitting
- **No Hyperparameter Tuning**: Good default parameters

## ✨ Features

- Streamlit web interface
- Native categorical feature support
- Fast training with GPU acceleration
- Automatic handling of missing values
- Built-in regularization
- Feature importance analysis

## 🛠️ Technologies

- Python 3.8+
- Streamlit
- CatBoost - Gradient boosting library
- scikit-learn - Preprocessing and metrics
- pandas, numpy

## 📦 Dataset

Customer segmentation with mixed data types (categorical and numerical).

**Data Files:**
- `data/Train.csv`
- `data/Test.csv`

## 🚀 Installation

```bash
cd Supervised_Learning/Classification/CatBoost
pip install -r requirements.txt
```

## 💻 Usage

```bash
streamlit run Home.py
```

## 📁 Project Structure

```
CatBoost/
├── Home.py          # Streamlit app
├── utils.py         # CatBoost implementation
├── data/            # Datasets
└── models/          # Trained models
```

## 📈 Model Performance

- **High Accuracy**: State-of-the-art performance
- **Fast Inference**: Optimized prediction speed
- **Categorical Features**: Handles them natively
- **Regularization**: Built-in overfitting prevention

## 🎯 Use Cases

- Customer segmentation
- Click-through rate prediction
- Recommendation systems
- Financial forecasting
- Any task with categorical features

## ⚙️ Advantages

- No need for one-hot encoding
- Handles missing values automatically
- Fast training and inference
- Excellent default parameters

---

**Developed by MEB**
