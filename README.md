# 🐾 Veterinary Disease Prediction using NLP  

AI-powered veterinary disease prediction using NLP and BERT - helping identify animal illnesses from symptoms with better accuracy.
This project predicts whether an animal disease is **dangerous or not** based on veterinary records using **Natural Language Processing (NLP)** and **Deep Learning (BERT)**.  

---

## 📂 Dataset  
- `season_priority_veterinary_records.csv`  
- Columns:  
  - `Animal`  
  - `Symptoms`  
  - `Diagnosis`  
  - `Treatment`  
  - `Season`  
  - `Dangerous` (Yes/No → Target variable)  

The text columns are **combined into a single input** for NLP-based classification.  

---

## ⚙️ Models Implemented  
1. **BERT (Transformers)** → Fine-tuned `bert-base-uncased` for text classification.  
2. **Random Forest (Scikit-learn)** → Baseline ML model.  
3. **K-Nearest Neighbors (KNN)** → Another baseline ML model.  

---

## 🚀 Project Workflow  

### 🔹 Data Preprocessing  
- Combine multiple textual fields into one input string.  
- Tokenize using HuggingFace **BERT tokenizer**.  
- Pad & truncate sequences to handle different input lengths.  

### 🔹 Model Training  
- Fine-tune **BERT model** using **PyTorch** and **Transformers library**.  
- Use **AdamW optimizer** with learning rate scheduling.  
- Train on **train/test split** of veterinary dataset.  

### 🔹 Evaluation  
- Evaluate predictions with:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
  - Classification Report  

### 🔹 Model Saving  
- Trained **BERT model & tokenizer** are saved locally for reuse.  

---

## 💻 Tech Stack  

- **Python**  
- **Pandas, NumPy** (data preprocessing)  
- **PyTorch** (deep learning framework)  
- **HuggingFace Transformers** (`bert-base-uncased`)  
- **Scikit-learn** (Random Forest, KNN, metrics)  
- **Matplotlib** (for any visualization/plots)  

---

## 📦 Installation  

Clone the repo:  
```bash
git clone https://github.com/Meenakshi2705/veterinary-disease-prediction.git

