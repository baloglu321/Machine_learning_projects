# LightGBM Customer Segmentation

LightGBM (Light Gradient Boosting Machine) implementation for high-performance customer segmentation. A fast, distributed, high-performance gradient boosting framework.

## 📊 Algorithm Overview

**LightGBM** is a gradient boosting framework developed by Microsoft that uses tree-based learning algorithms. It's designed for distributed and efficient training, especially with large datasets.

### Key Characteristics:
- **Faster Training**: Histogram-based algorithm
- **Lower Memory Usage**: Optimized memory consumption
- **Better Accuracy**: Leaf-wise tree growth
- **Parallel Learning**: GPU and distributed support
- **Large Dataset Support**: Handles millions of instances
- **Categorical Feature Support**: Native handling

## ✨ Features

- High-performance gradient boosting
- Fast training and prediction
- Streamlit web interface
- Memory-efficient implementation
- GPU acceleration support
- Automatic categorical feature handling

## 🛠️ Technologies

- Python 3.8+
- Streamlit
- LightGBM - Gradient boosting framework
- scikit-learn - Preprocessing
- pandas, numpy

## 📦 Dataset

Customer segmentation dataset.

**Data Files:**
- `data/Train.csv`
- `data/Test.csv`

## 🚀 Installation

```bash
cd Supervised_Learning/Classification/LightGBM
pip install -r requirements.txt
```

## 💻 Usage

```bash
streamlit run Home.py
```

## 📁 Project Structure

```
LightGBM/
├── Home.py          # Streamlit app
├── utils.py         # LightGBM implementation
├── data/            # Datasets
└── models/          # Trained models
```

## 📈 Model Performance

- **High Speed**: Faster than traditional GBDT
- **High Accuracy**: Leaf-wise growth strategy
- **Low Memory**: Histogram-based algorithm
- **Scalable**: Handles large datasets efficiently

## 🎯 Use Cases

- Customer segmentation
- Click prediction
- Ranking tasks
- Financial forecasting
- Large-scale classification

## ⚙️ Advantages Over Traditional GBM

- **Speed**: 20x faster training
- **Memory**: Lower memory consumption
- **Accuracy**: Better performance on large datasets
- **Scalability**: Distributed and GPU support

---

**Developed by MEB**
