# AdaBoost Customer Segmentation

AdaBoost (Adaptive Boosting) implementation for customer segmentation with Streamlit interface. A powerful ensemble method that combines multiple weak learners into a strong classifier.

## 📊 Algorithm Overview

**AdaBoost** is an ensemble learning method that combines multiple weak classifiers to create a strong classifier. It adaptively adjusts the weights of incorrectly classified instances, focusing on difficult cases in subsequent iterations.

### Key Characteristics:
- **Adaptive Learning**: Focuses on misclassified samples
- **Ensemble Method**: Combines weak learners
- **Weight Adjustment**: Updates sample weights iteratively
- **Versatile**: Works with various base estimators
- **Reduces Bias**: Improves model accuracy

## ✨ Features

- Interactive Streamlit dashboard
- Adaptive boosting classification
- Real-time predictions
- Model performance metrics
- Feature importance visualization
- Automated model training

## 🛠️ Technologies

- Python 3.8+
- Streamlit
- scikit-learn - AdaBoostClassifier
- pandas, numpy
- pickle

## 📦 Dataset

Customer segmentation dataset with demographic and behavioral attributes.

**Data Files:**
- `data/Train.csv`
- `data/Test.csv`

## 🚀 Installation

```bash
cd Supervised_Learning/Classification/AdaBoost
pip install -r requirements.txt
```

## 💻 Usage

```bash
streamlit run Home.py
```

## 📁 Project Structure

```
AdaBoost/
├── Home.py          # Streamlit interface
├── utils.py         # AdaBoost model logic
├── data/            # Datasets
└── models/          # Saved models
```

## 📈 Model Performance

- **Accuracy Score**: Overall classification accuracy
- **Ensemble Size**: Number of weak learners
- **Learning Rate**: Weight update factor
- **Base Estimator**: Usually Decision TreeClassifier

## 🎯 Use Cases

- Customer segmentation
- Fraud detection
- Face detection
- Text classification
- Medical diagnosis

## ⚙️ Hyperparameters

- `n_estimators`: Number of weak learners
- `learning_rate`: Weight applied to each classifier
- `base_estimator`: Weak learner algorithm

---

**Developed by MEB**
