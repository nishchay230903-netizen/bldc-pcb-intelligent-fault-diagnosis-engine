# 🔧 BLDC PCB Intelligent Fault Diagnosis Engine

An AI-powered fault diagnosis system for BLDC fan driver PCBs using similarity-based retrieval techniques.

---

## 🚀 Overview

This project was developed to analyze and diagnose faults in BLDC (Brushless DC) fan driver PCBs based on electrical parameters and observed symptoms.

It uses a machine learning approach based on **TF-IDF vectorization and cosine similarity** to match input conditions with known fault cases.

---

## 🧠 Key Features

* 🔍 Multi-fault prediction with ranking
* 📊 Confidence score for each predicted fault
* ⚡ Handles real-world PCB failure scenarios
* 🖥️ Interactive UI using Streamlit
* 📁 Lightweight and fast (no heavy model required)

---

## ⚙️ Tech Stack

* Python
* Streamlit
* Scikit-learn (TF-IDF, Cosine Similarity)
* Pandas, NumPy
* JSON-based dataset

---

## 📊 How It Works

1. User inputs PCB parameters (current, voltage, wattage, symptoms)
2. Input is converted into a text-based feature vector using TF-IDF
3. Cosine similarity is calculated against historical fault data
4. Top matching faults are returned with confidence scores

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📸 Demo

(Add screenshot here after running the app)

---

## 📌 Real-World Application

* Industrial quality control (QC)
* Fault analysis in electronics manufacturing
* Predictive maintenance systems
* Internship project at Orient Electric

---

## 🎯 Project Highlights

* Built using real PCB fault observations
* Designed for practical deployment scenarios
* Efficient alternative to heavy ML models
