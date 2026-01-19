# GBM (Gradient Boosting Machine) Customer Segmentation

Gradient Boosting Machine implementation for customer segmentation. A powerful ensemble technique that builds models sequentially to correct errors of previous models.

## 📊 Algorithm Overview

**Gradient Boosting Machine (GBM)** is an ensemble learning technique that builds an additive model in a forward stage-wise fashion. It optimizes arbitrary differentiable loss functions through gradient descent.

### Key Characteristics:
- **Sequential Learning**: Each model corrects previous errors
- **Gradient Descent**: Optimizes loss function directly
- **Flexible**: Supports various loss functions
- **High Performance**: Excellent predictive accuracy
- **Feature Importance**: Identifies key predictors
- **Handles Mixed Data**: Numerical and categorical features

## ✨ Features

- Interactive Streamlit interface
- Sequential ensemble learning
- Customizable loss functions
- Real-time customer segmentation
- Model performance tracking
- Feature importance visualization

## 🛠️ Technologies

- Python 3.8+
- Streamlit
- scikit-learn - GradientBoostingClassifier
- pandas, numpy
- matplotlib, seaborn

## 📦 Dataset

Customer segmentation dataset.

**Data Files:**
- `data/Train.csv`
- `data/Test.csv`

## 🚀 Installation

```bash
cd Supervised_Learning/Classification/GBM
pip install -r requirements.txt
```

## 💻 Usage

```bash
streamlit run Home.py
```

## 📁 Project Structure

```
GBM/
├── Home.py          # Streamlit application
├── utils.py         # GBM model implementation
├── data/            # Training and test data
└── models/          # Saved models
```

## 📈 Model Performance

- **Accuracy**: High classification accuracy
- **Learning Rate**: Controls contribution of each tree
- **N_estimators**: Number of boosting stages
- **Max Depth**: Individual tree complexity

## 🎯 Use Cases

- Customer segmentation
- Credit risk modeling
- Churn prediction
- Ranking problems
- Click-through rate prediction

## ⚙️ Hyperparameters

- `n_estimators`: Number of boosting stages (100-1000)
- `learning_rate`: Shrinkage parameter (0.01-0.3)
- `max_depth`: Maximum tree depth (3-10)
- `subsample`: Fraction of samples for fitting

---

**Developed by MEB**
