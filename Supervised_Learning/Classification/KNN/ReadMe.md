# K-Nearest Neighbors (KNN) Customer Segmentation

K-Nearest Neighbors classification for customer segmentation with Streamlit. A simple yet powerful instance-based learning algorithm that classifies based on similarity to neighboring data points.

## 📊 Algorithm Overview

**K-Nearest Neighbors (KNN)** is a non-parametric, lazy learning algorithm that classifies instances based on the majority class among the k nearest neighbors in the feature space.

### Key Characteristics:
- **Instance-Based**: Stores all training data
- **Non-Parametric**: Makes no assumptions about data distribution
- **Simple**: Easy to understand and implement
- **Versatile**: Works for classification and regression
- **Distance-Based**: Uses similarity metrics (Euclidean, Manhattan, etc.)
- **Lazy Learning**: No explicit training phase

## ✨ Features

- Interactive Streamlit dashboard
- Distance-based classification
- Configurable k parameter
- Real-time predictions
- Multiple distance metrics
- Visualization of decision boundaries

## 🛠️ Technologies

- Python 3.8+
- Streamlit
- scikit-learn - KNeighborsClassifier
- pandas, numpy
- matplotlib

## 📦 Dataset

Customer segmentation dataset.

**Data Files:**
- `data/Train.csv`
- `data/Test.csv`

## 🚀 Installation

```bash
cd Supervised_Learning/Classification/KNN
pip install -r requirements.txt
```

## 💻 Usage

```bash
streamlit run Home.py
```

## 📁 Project Structure

```
KNN/
├── Home.py          # Streamlit interface
├── utils.py         # KNN implementation
├── data/            # Datasets
└── models/          # Saved models
```

## 📈 Model Performance

- **Accuracy**: Depends on k value and distance metric
- **K Selection**: Typically use odd numbers to avoid ties
- **Feature Scaling**: Critical for good performance
- **Distance Metric**: Euclidean, Manhattan, Minkowski

## 🎯 Use Cases

- Customer segmentation
- Pattern recognition
- Recommendation systems
- Image classification
- Anomaly detection

## ⚙️ Hyperparameters

- `n_neighbors`: Number of neighbors (k)
- `metric`: Distance calculation method
- `weights`: Uniform or distance-based weighting

---

**Developed by MEB**
