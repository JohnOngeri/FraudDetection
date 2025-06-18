# Fraud Detection in Mobile Money Transactions
## Machine Learning Models with Optimization Techniques

### Project Overview
This project implements and compares various neural network models for detecting fraudulent transactions in mobile money systems. The analysis focuses on the impact of different optimization techniques, regularization methods, and hyperparameter configurations on model performance.

### Dataset Description
**Source**: [Kaggle - Fraud Detection in Mobile Transactions](https://www.kaggle.com/code/tomaszurban/fraud-detection-in-mobile-transactions/input)

**MY VIDEO:**
[https://drive.google.com/file/d/1S9hjkY_Tbhd2gF95J88Wttz8IXeEwVih/view?usp=sharing](url)

**Problem Statement**: Binary classification to predict whether a mobile money transaction is fraudulent or legitimate.

**Dataset Characteristics**:
- **Total Transactions**: 1,048,575
- **Fraudulent Transactions**: 1,142 (0.109%)
- **Non-fraudulent Transactions**: 1,047,433 (99.891%)
- **Features**: 11 (after preprocessing)
- **Class Imbalance**: Extreme (1:918 ratio)

**Key Features**:
- `step`: Time unit (1 step = 1 hour)
- `type`: Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER)
- `amount`: Transaction amount in local currency
- `oldbalanceOrg/newbalanceOrig`: Account balances before/after transaction
- `oldbalanceDest/newbalanceDest`: Recipient balances before/after transaction
- `isFraud`: Target variable (0=legitimate, 1=fraudulent)

### Data Preprocessing Pipeline

#### 1. Data Cleaning
- Removed non-informative ID columns (`nameOrig`, `nameDest`)
- One-hot encoded categorical variables (`type`)
- Verified all features are numeric

#### 2. Train-Validation-Test Split
- **Training**: 60% (629,145 samples)
- **Validation**: 20% (209,715 samples) 
- **Test**: 20% (209,715 samples)
- Stratified sampling to maintain class distribution

#### 3. Class Imbalance Handling
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) on training set only
- Balanced training set: 628,459 samples per class
- Validation and test sets kept imbalanced to reflect real-world conditions

#### 4. Feature Scaling
- **StandardScaler** applied to normalize all features
- Fitted on training data, applied to validation and test sets


## 📊 Performance Comparison Table

| Instance                | Optimizer           | Regularization                 | Learning Rate | Early Stopping       | Epochs | Accuracy   | Precision | Recall  | F1-Score | Validation Loss |
|-------------------------|---------------------|--------------------------------|----------------|-----------------------|--------|------------|-----------|---------|----------|------------------|
| 1 (Baseline)            | Adam (default)      | None                           | 0.001          | No                    | 10     | 99.02%     | 0.093     | 0.930   | 0.170    | 0.0287           |
| 2                       | RMSprop             | L2 (λ=0.01)                    | 0.0005         | Yes (patience=5)      | 20     | 99.00%     | 0.088     | 0.877   | 0.160    | 0.0800           |
| 3                       | Adam                | Dropout (30%)                  | 0.0003         | Yes (patience=5)      | 12     | **99.11%** | 0.102     | **0.917** | **0.183** | 0.0318           |
| 4                       | SGD                 | L1 (λ=0.001) + Dropout (30%)   | 0.01           | Yes (patience=5)      | 15     | **99.74%** | **0.264** | 0.759   | **0.392** | 0.0893           |
| 5 (Logistic Regression) | liblinear (L1)      | L1 Penalty                     | N/A            | N/A                   | N/A    | 95.00%     | 0.020     | **0.950** | 0.040    | N/A              |


## 🏗️ Model Architectures

| Instance                | Architecture Description             | Number of Layers |
|-------------------------|--------------------------------------|------------------|
| 1 (Baseline)            | 3 Dense layers (64 → 32 → 1)         | 3                |
| 2                       | 4 Dense layers (128 → 64 → 32 → 1)   | 4                |
| 3                       | 4 Dense layers (128 → 64 → 32 → 1) + Dropout (30%) | 4                |
| 4                       | 4 Dense layers (128 → 64 → 32 → 1) + L1 (λ=0.001) + Dropout (30%) | 4                |
| 5 (Logistic Regression) | No hidden layers (linear classifier) | 0                |



#### Test Set Performance (Best Model - Instance 4)
- **Accuracy**: 99.01%
- **Precision**: 8.60%
- **Recall**: 84.21%
- **F1-Score**: 15.61%

### Key Findings and Analysis

#### 1. Class Imbalance Impact
- High accuracy (>99%) across all models due to dataset imbalance
- **Recall** is critical metric for fraud detection (catching actual frauds)
- **Precision** remains low due to extreme class imbalance (many false positives)

#### 2. Optimization Technique Effectiveness

**L2 Regularization (Instance 2)**:
- ✅ Prevented overfitting effectively
- ✅ Stable training curves
- ❌ Conservative performance, lower precision

**Dropout Regularization (Instance 3)**:
- ✅ Best generalization capability
- ✅ Improved recall while maintaining precision
- ✅ Robust to overfitting

**Combined L1 + Dropout (Instance 4)**:
- ✅ **Highest precision** (26.41%) while maintaining good recall
- ✅ **Best F1-score** (39.18%) on validation set
- ✅ SGD optimizer with higher learning rate worked well with regularization

#### 3. Optimizer Comparison
- **Adam**: Fast convergence, good default choice
- **RMSprop**: Stable training with L2 regularization
- **SGD**: Surprisingly effective with proper regularization and learning rate

#### 4. Training Efficiency
- **Early Stopping** significantly reduced training time (12-20 epochs vs 50)
- **Learning Rate Tuning** crucial for SGD performance
- **Batch Size** of 64 provided good balance between speed and stability

### Business Impact Analysis

#### Model Selection Rationale
**Instance 4 (SGD + L1 + Dropout)** selected as production model because:

1. **Highest Precision (26.41%)**: Reduces false alarms, minimizing customer friction
2. **Good Recall (75.88%)**: Catches 3 out of 4 fraudulent transactions
3. **Best F1-Score**: Optimal balance between precision and recall
4. **Computational Efficiency**: SGD is memory-efficient for large-scale deployment

#### Real-World Implications
- **True Positive Rate**: 84.21% of frauds detected
- **False Positive Rate**: ~0.8% of legitimate transactions flagged
- **Cost-Benefit**: Model prevents significant fraud losses while maintaining user experience

### Technical Implementation

#### Libraries and Dependencies
```python
# Core ML and Data Processing
pandas, numpy, scikit-learn
tensorflow, keras
imblearn (SMOTE)

# Visualization
matplotlib, seaborn

# Model Persistence
joblib

# Architecture Visualization
pydot, graphviz
```

#### Model Persistence
All trained models saved in `saved_models/` directory:
- `nn_instance1_basic.keras`
- `nn_instance2_optimized.keras` 
- `nn_instance3_optimized.keras`
- `nn_instance4_optimized.keras`
- `logistic_regression_model.pkl`

### Future Improvements

#### 1. Advanced Techniques
- **Ensemble Methods**: Combine multiple models for better performance
- - **Threshold Optimization**: Tune decision threshold for business requirements

#### 2. Feature Engineering
- **Temporal Features**: Transaction patterns over time
- **Behavioral Features**: User transaction history
- **Network Analysis**: Relationship between accounts

#### 3. Model Monitoring
- **Drift Detection**: Monitor for changes in transaction patterns
- - **Feedback Loop**: Incorporate fraud analyst feedback

### Conclusion

This project successfully demonstrates the impact of optimization and regularization techniques on neural network performance for fraud detection. The combination of **SGD optimizer with L1 regularization and Dropout** (Instance 4) achieved the best balance between detecting fraudulent transactions and minimizing false alarms.

**Key Takeaways**:
1. **Regularization is crucial** for fraud detection models due to class imbalance
2. **Combined regularization techniques** (L1 + Dropout) outperform single methods
3. **Proper hyperparameter tuning** can make traditional optimizers (SGD) competitive
4. **Business context matters** - optimize for the right metric (F1-score for fraud detection)

The final model provides a solid foundation for a production fraud detection system, with clear paths for future enhancement and monitoring.

**---


```
FraudDetection/
│
├── fraud.csv                         # Dataset
├── saved_models/                     # Folder with all saved Keras and sklearn models
│   ├── logistic_regression_model.pkl
│   ├── nn_instance1_basic.keras
│   ├── nn_instance2_optimized.keras
│   ├── nn_instance3_optimized.keras
│   └── nn_instance4_optimized.keras
├── model_architectures/              # Folder with model architecture images (optional)
│   ├── Instance1_Basic_Neural_Network_architecture.png
│   └── ...
├── Summative_Intro_to_ml_[John Ongeri Ouma].ipynb  # Main notebook
├── README.md
└── 
```

---

### 📊 Best Model

✅ **Instance 4** was the best-performing model with the highest F1-Score:

-
Saved as:  
```bash
saved_models/nn_instance4_optimized.keras
```

---

### ▶️ How to Run the Notebook

1. Clone the repo or open the notebook in Google Colab:

```bash
https://github.com/JohnOngeri/FraudDetection.git
```

2. Open the notebook `Summative_Intro_to_ml_[John Ongeri Ouma].ipynb`.

3. Run all cells sequentially, starting from data preprocessing to model training.

---

### 🧪 Making Predictions with the Best Model

To load and make predictions using the best saved model:

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the best model (Instance 4)
model = load_model('saved_models/nn_instance4_optimized.keras')

# Predict on test set
y_pred = (model.predict(X_test_scaled) > 0.5).astype('int32')
```

---


- If you're visualizing model architectures, install Graphviz and ensure it’s in your system PATH.

---

### 🙌 Author

- **John Ongeri Ouma**  
  [GitHub](https://github.com/JohnOngeri) | j.ouma@alustudent.com

