# SVM (Support Vector Machine) Customer Segmentation

Support Vector Machine implementation for customer segmentation. A powerful algorithm that finds optimal hyperplanes for classification.

## 📊 Algorithm Overview

**Support Vector Machine (SVM)** is a supervised learning algorithm that finds the optimal hyperplane that maximally separates different classes in high-dimensional space.

### Key Characteristics:
- **Maximum Margin**: Finds optimal decision boundary
- **Kernel Trick**: Handles non-linear relationships
- **Support Vectors**: Only key points influence model
- **Effective in High Dimensions**: Works well with many features
- **Robust**: Less prone to overfitting in high dimensions
- **Versatile**: Multiple kernel functions (linear, RBF, polynomial)

## ✨ Features

- Interactive Streamlit interface
- Multiple kernel options (linear, RBF, polynomial)
- Optimal hyperplane separation
- Support vector identification
- Real-time classification
- Margin visualization

## 🛠️ Technologies

- Python 3.8+
- Streamlit
- scikit-learn - SVC (Support Vector Classifier)
- pandas, numpy
- matplotlib

## 📦 Dataset

Customer segmentation dataset.

**Data Files:**
- `data/Train.csv`
- `data/Test.csv`

## 🚀 Installation

```bash
cd Supervised_Learning/Classification/SVM
pip install -r requirements.txt
```

## 💻 Usage

```bash
streamlit run Home.py
```

## 📁 Project Structure

```
SVM/
├── Home.py          # Streamlit application
├── utils.py         # SVM implementation
├── data/            # Datasets
└── models/          # Trained models
```

## 📈 Model Performance

- **Accuracy**: Excellent with proper kernel selection
- **Kernel Choice**: Linear, RBF, polynomial, sigmoid
- **C Parameter**: Regularization strength
- **Gamma**: Kernel coefficient (for RBF/polynomial)

## 🎯 Use Cases

- Customer segmentation
- Image classification
- Text classification
- Bioinformatics
- Face detection

## ⚙️ Hyperparameters

- `kernel`: Type of kernel function
- `C`: Regularization parameter
- `gamma`: Kernel coefficient
- `degree`: Polynomial degree (for poly kernel)

---

**Developed by MEB**
