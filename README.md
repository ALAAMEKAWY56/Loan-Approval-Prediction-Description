# 💳 Loan Approval Prediction

## 📌 Project Overview
This project predicts whether a loan application will be approved based on applicant information such as income, employment, and credit history.  
The dataset is **imbalanced**, so we applied **SMOTE** (Synthetic Minority Over-sampling Technique) to improve model performance.  

Two models — **Logistic Regression** (baseline) and **Decision Tree Classifier** (tuned) — were built and evaluated.

---

## 🚀 Key Highlights
- **Goal:** Predict loan approval status (binary classification).  
- **Dataset:** Loan Approval Prediction Dataset (Kaggle).  
- **Class Imbalance:** Addressed using SMOTE.  
- **Models Used:** Logistic Regression, Decision Tree Classifier.  
- **Evaluation Metrics:** Accuracy, Precision, Recall, and F1 Score.  
- **Visualization:**  
  - Class distribution **before and after SMOTE**.  
  - Feature distributions and correlation heatmaps.  

---

## 📊 Performance Results

| **Metric**   | **Logistic Regression** | **Decision Tree** |
|--------------|-------------------------|--------------------|
| Accuracy     | 0.7845                  | 0.9707             |
| Precision    | 0.7233                  | 0.9592             |
| Recall       | 0.6824                  | 0.9623             |
| F1 Score     | 0.7023                  | 0.9608             |

The **Decision Tree Classifier** significantly outperformed Logistic Regression, achieving nearly **97% accuracy** and balanced precision-recall scores.

---

## 🛠 Tools & Libraries
- **Python**  
- **Pandas, NumPy**  
- **Scikit-learn** (Logistic Regression, Decision Tree, metrics)  
- **Imbalanced-learn** (SMOTE)  
- **Matplotlib, Seaborn** (Visualizations)

---

## 📈 Workflow
1. **Data Preprocessing:**  
   - Missing value imputation.  
   - Encoding categorical features.  
2. **EDA (Exploratory Data Analysis):**  
   - Visualized class imbalance and feature distributions.  
3. **Class Imbalance Handling:**  
   - Applied SMOTE and visualized **before vs. after** distributions.  
4. **Model Training:**  
   - Logistic Regression (baseline).  
   - Decision Tree (optimized with hyperparameters).  
5. **Model Evaluation:**  
   - Compared metrics (Accuracy, Precision, Recall, F1).  
   - Confusion matrix analysis.

---

## 🧩 How to Run
1. Clone the repository:

    git clone https://github.com/ALAAMEKAWY56/loan-approval-prediction.git

2. Install dependencies:

    pip install -r requirements.txt

3. Run The NoteBook:

    jupyter notebook Loan_Approval_Prediction.ipynb

---

## 📘 License

This project is licensed for educational and personal use.

---

## 🙋‍♀️ Author

**Alaa Mohamed Mekawi**  
Computer Engineer | Data & AI Enthusiast

---
