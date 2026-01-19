# Decision Trees Customer Segmentation

A Streamlit application implementing Decision Tree classification for customer segmentation. Decision Trees provide interpretable, tree-based learning for classification tasks.

## 📊 Algorithm Overview

**Decision Trees** are non-parametric supervised learning methods used for classification and regression. The algorithm creates a model that predicts the target by learning simple decision rules inferred from data features.

### Key Characteristics:
- **Interpretable**: Easy to understand and visualize
- **No Feature Scaling Required**: Works with raw feature values
- **Handles Non-linear Relationships**: Captures complex patterns
- **Feature Importance**: Identifies most influential features
- **Works with Mixed Data**: Handles both numerical and categorical features

## ✨ Features

- Interactive Streamlit web interface
- Visual decision tree representation
- Feature importance analysis
- Real-time customer segment predictions
- No preprocessing required for tree algorithms
- Model persistence and reusability

## 🛠️ Technologies

- Python 3.8+
- Streamlit - Web framework
- scikit-learn - DecisionTreeClassifier
- pandas - Data manipulation
- matplotlib - Visualizations (optional)

## 📦 Dataset

Customer segmentation data with demographic and behavioral features.

**Data Files:**
- `data/Train.csv`
- `data/Test.csv`

## 🚀 Installation

```bash
cd Supervised_Learning/Classification/DecisionTrees
pip install -r requirements.txt
```

## 💻 Usage

```bash
streamlit run Home.py
```

## 📁 Project Structure

```
DecisionTrees/
├── Home.py          # Streamlit app
├── utils.py         # Decision Tree implementation
├── data/            # Datasets
└── models/          # Trained models
```

## 📈 Model Performance

- **Accuracy**: Classification accuracy on test set
- **Feature Importance**: Gini importance scores
- **Tree Depth**: Configurable maximum depth
- **Pruning**: Prevents overfitting

## 🎯 Use Cases

- Customer segmentation
- Credit approval decisions
- Medical diagnosis
- Fraud detection
- Product recommendations

---

**Developed by MEB**
