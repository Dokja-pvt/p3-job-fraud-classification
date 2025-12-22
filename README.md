# Fake Job Posting Detection System

## Abstract

The Fake Job Posting Detection System is an academic machine learning project designed to identify fraudulent job advertisements using Natural Language Processing (NLP) techniques. The system analyzes textual job descriptions and classifies them as genuine or fraudulent, helping reduce the risks associated with online recruitment scams. The project emphasizes balanced evaluation on imbalanced data and demonstrates practical deployment through a web-based interface.

---

## Project Overview

Online recruitment platforms have simplified job searching but have also enabled the rise of fake job postings that mislead applicants through deceptive language and unrealistic offers. Manual verification methods are time-consuming and unreliable due to the high volume of job listings. This project addresses the problem by applying supervised machine learning and NLP techniques to automatically detect fraudulent job postings.

The trained model is deployed using Streamlit, allowing users to interactively submit job descriptions and receive real-time predictions.

---

## Objectives

* To preprocess and analyze job posting text using NLP techniques
* To extract meaningful textual features using TF-IDF
* To train and evaluate a machine learning classification model
* To address class imbalance using appropriate evaluation metrics
* To deploy the trained model through a user-friendly web interface

---

## Technology Stack

* **Programming Language:** Python
* **Libraries:** Scikit-learn, Pandas, NumPy
* **NLP Techniques:** Text preprocessing, TF-IDF vectorization
* **Model Persistence:** Joblib
* **Web Framework:** Streamlit

---

## Machine Learning Workflow

1. Dataset Collection
2. Exploratory Data Analysis (EDA)
3. Text Preprocessing
4. Feature Extraction using TF-IDF
5. Model Training and Evaluation
6. Model Saving using Joblib
7. Deployment via Streamlit

---

## Performance Metrics

* **Primary Metric:** F1-Score = **0.6397**
* **Accuracy:** ~94% (not prioritized due to class imbalance)
* **Evaluation Tools:** Confusion Matrix, Classification Report

The F1-score was selected as the primary evaluation metric as it provides a balanced measure of precision and recall for imbalanced datasets.

---

## Application Features

* Automated classification of job postings as Real or Fake
* NLP-based text preprocessing
* Real-time prediction through a web interface
* Lightweight and efficient deployment

---

## Project Structure

```
Fake-Job-Posting-Detection/
│
├── data/                  # Dataset files
├── notebooks/             # Model training and experimentation
├── app.py                 # Streamlit application
├── model.pkl              # Trained machine learning model
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## How to Run the Project

1. Clone the repository:

```
git clone https://github.com/your-username/fake-job-posting-detection.git
```

2. Navigate to the project directory:

```
cd fake-job-posting-detection
```

3. Install required dependencies:

```
pip install -r requirements.txt
```

4. Run the Streamlit application:

```
streamlit run app.py
```

---

## Future Scope

* Integration of deep learning models such as LSTM and BERT
* Real-time integration with job portals
* Multilingual job posting analysis
* Browser extension for instant fraud detection

---

## References

* Vidros et al., *Automatic Detection of Online Recruitment Frauds*, 2017
* Almusharraf, *Detecting Fake Job Postings Using Machine Learning*, 2020
* Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers*, 2019
* UCI Machine Learning Repository

---

## Authors

* *Furqan Mohammad*
* Department of Computer Applications
* Jagran Lakecity University, Bhopal
