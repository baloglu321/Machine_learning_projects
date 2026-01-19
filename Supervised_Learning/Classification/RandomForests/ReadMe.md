# Random Forests Customer Segmentation

Random Forest ensemble learning for customer segmentation with Streamlit. A powerful ensemble method that combines multiple decision trees for robust classification.

## 📊 Algorithm Overview

**Random Forests** is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of classes for classification. It introduces randomness in both feature selection and data sampling.

### Key Characteristics:
- **Ensemble Method**: Combines multiple decision trees
- **Bagging**: Bootstrap aggregating for variance reduction
- **Feature Randomness**: Random feature subset at each split
- **Robust**: Resistant to overfitting
- **Feature Importance**: Identifies key predictors
- **Handles Missing Values**: Built-in imputation

## ✨ Features

- Interactive Streamlit dashboard
- Ensemble of decision trees
- Feature importance ranking
- Out-of-bag error estimation
- Parallelized training
- Robust predictions

## 🛠️ Technologies

- Python 3.8+
- Streamlit
- scikit-learn - RandomForestClassifier
- pandas, numpy
- matplotlib, seaborn

## 📦 Dataset

Customer segmentation dataset.

**Data Files:**
- `data/Train.csv`
- `data/Test.csv`

## 🚀 Installation

```bash
cd Supervised_Learning/Classification/RandomForests
pip install -r requirements.txt
```

## 💻 Usage

```bash
streamlit run Home.py
```

## 📁 Project Structure

```
RandomForests/
├── Home.py          # Streamlit interface
├── utils.py         # Random Forest implementation
├── data/            # Datasets
└── models/          # Trained models
```

## 📈 Model Performance

- **High Accuracy**: Ensemble reduces variance
- **Feature Importance**: Gini or entropy-based
- **OOB Score**: Out-of-bag error estimate
- **Parallelization**: Fast training on multi-core CPUs

## 🎯 Use Cases

- Customer segmentation
- Feature selection
- Anomaly detection
- Medical diagnosis
- Credit risk assessment

## ⚙️ Hyperparameters

- `n_estimators`: Number of trees (100-1000)
- `max_depth`: Maximum tree depth
- `max_features`: Features considered at each split
- `min_samples_split`: Minimum samples to split node

---

**Developed by MEB**
