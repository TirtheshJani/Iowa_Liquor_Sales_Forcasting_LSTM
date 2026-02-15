# 🍷 Iowa Liquor Sales Forecasting with LSTM

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

> **Deep Learning for Retail Demand Forecasting**  
> LSTM neural networks applied to predict Iowa liquor sales using historical time series data.

---

## 📊 Project Overview

This project implements **Long Short-Term Memory (LSTM)** networks to forecast liquor sales for the state of Iowa. The model analyzes historical sales patterns to predict future demand, enabling better inventory management and business planning.

### Why LSTM for Time Series?

LSTM networks are particularly effective for time series forecasting because they:
- 🧠 **Remember long-term patterns** through their gating mechanisms
- 📈 **Handle sequential dependencies** better than traditional models
- 🔄 **Avoid vanishing gradient** problems of standard RNNs
- ⏱️ **Capture temporal dynamics** in sales data

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | TensorFlow, Keras |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |

---

## 📈 Methodology

### 1. Data Preprocessing

#### Scaling
```python
# MinMax scaling to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(sales_data)
```

#### Sequence Creation
- **Input sequence:** Past N days of sales data
- **Output:** Predicted sales for day N+1
- **Window size:** 7 days (weekly patterns)

### 2. Model Architecture

```
Input Layer (sequence_length, features)
    ↓
LSTM Layer 1 (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (50 units)
    ↓
Dropout (0.2)
    ↓
Dense Layer (25 units)
    ↓
Output Layer (1 unit)
```

### 3. Training Configuration

| Parameter | Value |
|-----------|-------|
| **Loss Function** | Mean Squared Error (MSE) |
| **Optimizer** | Adam |
| **Batch Size** | 32 |
| **Epochs** | 100 (with early stopping) |
| **Validation Split** | 20% |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Installation
```bash
# Clone the repository
git clone https://github.com/TirtheshJani/Iowa_Liquor_Sales_Forcasting_LSTM.git

# Navigate to project
cd Iowa_Liquor_Sales_Forcasting_LSTM

# Launch Jupyter notebook
jupyter notebook "Final Model Application and result.ipynb"
```

---

## 📊 Results & Performance

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **MAPE** | Mean Absolute Percentage Error |

### Key Findings
- ✅ LSTM effectively captures seasonal patterns
- ✅ Model shows strong performance on validation data
- ✅ Early stopping prevents overfitting
- ✅ Proper scaling crucial for LSTM performance

---

## 📧 Contact

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tirthesh-jani)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/TirtheshJani)
