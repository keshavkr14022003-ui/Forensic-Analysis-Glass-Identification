# 🔬 Forensic Glass Analysis

> **Machine Learning Based Glass Type Classification using Refractive Index and Chemical Composition**

An interactive **Streamlit-based forensic glass analysis platform** that uses machine learning to classify glass samples based on their physical and chemical properties.

The application provides an end-to-end workflow for entering glass sample measurements, generating predictions, exploring the dataset, evaluating model performance, and generating downloadable analysis reports.

---

## 🚀 Live Application

🌐 **Streamlit App:** *Add your deployed Streamlit URL here*

---

## 📌 Project Overview

Forensic glass fragments can provide useful physical and chemical characteristics for comparison and classification.

This project applies **Machine Learning** to classify a glass sample into one of the glass categories represented in the dataset.

The application takes **9 input features**:

* Refractive Index (RI)
* Sodium (Na)
* Magnesium (Mg)
* Aluminum (Al)
* Silicon (Si)
* Potassium (K)
* Calcium (Ca)
* Barium (Ba)
* Iron (Fe)

These features are passed through the saved scaler and machine-learning model to generate a predicted glass type.

---

## ✨ Features

### 🏠 Interactive Dashboard

* Project overview
* Dataset statistics
* Number of input features
* Observed glass classes
* Quick-start navigation
* Analysis workflow visualization

### 🔬 Glass Sample Analysis

Enter the physical and chemical properties of a glass sample and get:

* Predicted glass type
* Glass family and processing type
* Model probability
* Confidence indicator
* Prediction probability chart
* Input summary
* Prediction history
* Downloadable PDF analysis report

The application uses the saved scaler before sending the input to the trained model.

### 📊 Dataset Explorer

The application allows you to:

* View the dataset
* Filter by glass type
* Search values
* Select columns
* Download filtered data as CSV
* View class distribution
* Explore descriptive statistics
* Examine feature correlations
* Visualize individual feature distributions

### 🤖 Model Performance

The model evaluation section provides:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report
* Feature Importance
* Model benchmark comparison

The evaluation recreates the project's **80/20 train-test split with `random_state=42`**.

### 📚 Glass Type Reference

The application includes reference information for the dataset labels:

| Type | Glass Category   | Processing / Description      |
| ---- | ---------------- | ----------------------------- |
| 1    | Building Windows | Float Processed               |
| 2    | Building Windows | Non-Float Processed           |
| 3    | Vehicle Windows  | Float Processed               |
| 4    | Vehicle Windows  | Non-Float Processed           |
| 5    | Containers       | Bottles / Jars                |
| 6    | Tableware        | Drinking Glasses / Dishes     |
| 7    | Headlamps        | Automotive / Heavy-Duty Lamps |

The original dataset contains observed classes **1, 2, 3, 5, 6 and 7**; Type 4 is retained in the application's reference dictionary for completeness.

---

## 🧠 Machine Learning Models

Several machine-learning algorithms were evaluated during experimentation:

| Model                | Recorded Accuracy |
| -------------------- | ----------------: |
| 🥇 Gradient Boosting |        **86.05%** |
| Random Forest        |            83.72% |
| Extra Trees          |            81.40% |
| SVM                  |            72.09% |
| Logistic Regression  |            72.09% |
| KNN                  |            69.77% |

The **Gradient Boosting model** was selected for the application based on the recorded benchmark results.

> **Note:** These benchmark values are the recorded results from the supplied training notebook and are displayed by the application as experiment results.

---

## 🔄 Project Workflow

```text
Glass Sample
     ↓
Measure Physical & Chemical Properties
     ↓
Enter 9 Features
     ↓
Feature Scaling
     ↓
Gradient Boosting Model
     ↓
Glass Type Prediction
     ↓
Probability & Interpretation
     ↓
PDF Analysis Report
```

The application's documented workflow is:

**Measure → Enter → Scale → Predict → Interpret**.

---

## 🛠️ Technology Stack

| Category             | Technology                   |
| -------------------- | ---------------------------- |
| Frontend / UI        | Streamlit                    |
| Programming Language | Python                       |
| Data Processing      | Pandas, NumPy                |
| Machine Learning     | Scikit-learn                 |
| Model Storage        | Joblib                       |
| Visualization        | Streamlit Charts, Matplotlib |
| Report Generation    | ReportLab                    |

These technologies are used directly within the application.

---

## 📂 Project Structure

```text
Forensic-Glass-Analysis/
│
├── app.py
├── glass.csv
│
├── glass_prediction_model.pkl
├── glass_scaler.pkl
│
├── background.png
│
├── Glass_Type_Prediction.ipynb
├── requirements.txt
└── README.md
```

### Important Files

**`app.py`**
Main Streamlit application containing the dashboard, prediction system, dataset explorer, model evaluation, glass reference and report generation.

**`glass.csv`**
Dataset used by the application.

**`glass_prediction_model.pkl`**
Saved trained machine-learning model.

**`glass_scaler.pkl`**
Saved feature scaler used before prediction.

**`background.png`**
Background image used in the application's dashboard.

**`Glass_Type_Prediction.ipynb`**
Notebook containing the machine-learning experimentation and model comparison.

---

## 📦 Requirements

A typical `requirements.txt` for this project is:

```txt
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
reportlab
openpyxl
```

---

## 🧪 Using the Application

### Step 1 — Open **Analyze Sample**

Enter the measured values of:

```text
RI
Na
Mg
Al
Si
K
Ca
Ba
Fe
```

### Step 2 — Click **Analyze Sample**

The application scales the input and sends it to the saved machine-learning model.

### Step 3 — View the Result

The application displays:

* Predicted Type
* Glass category
* Processing type
* Highest model probability
* Probability distribution

### Step 4 — Generate Report

A PDF report can be generated containing the prediction, model information, sample properties and model probability.

---

## 📊 Model Evaluation

The application calculates:

* Accuracy
* Weighted Precision
* Weighted Recall
* Weighted F1 Score
* Confusion Matrix
* Classification Report

It also provides feature importance when the loaded model exposes `feature_importances_`.

---

## ⚠️ Important Disclaimer

This project is intended for **educational and analytical purposes**.

The predictions generated by the machine-learning model should **not** be treated as a standalone forensic conclusion.

Actual forensic glass examination should involve validated laboratory methods and expert interpretation. The application itself explicitly presents its predictions as an analytical aid rather than a replacement for laboratory examination.

---

## 🎯 Future Improvements

Possible improvements include:

* [ ] Add more glass samples to the dataset
* [ ] Improve class imbalance handling
* [ ] Hyperparameter tuning
* [ ] Cross-validation
* [ ] Automated model retraining
* [ ] SHAP-based model explainability
* [ ] Upload CSV for batch predictions
* [ ] Batch prediction report generation
* [ ] Authentication and user management
* [ ] Cloud deployment
* [ ] Prediction history database
* [ ] More advanced forensic visualizations

---

## 👨‍💻 Author

**Keshav Kumar**

B.Tech Computer Science & Engineering

---

## ⭐ If You Like This Project

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub!

---

### 📄 License

This project is intended for educational and academic use.
