# SoleusAI

AI-Powered Soleus Muscle Activation Optimization System

## Overview

SoleusAI is an intelligent biomechanical analysis platform designed to estimate soleus muscle activation using EMG-derived features and machine learning techniques.

The system combines machine learning, retrieval-augmented generation (RAG), and AI-driven coaching to assist researchers, physiotherapists, sports scientists, and rehabilitation professionals in understanding muscle activation behavior.

---

## Objectives

* Predict soleus muscle activation using EMG features.
* Provide intelligent resistance training recommendations.
* Build a research knowledge base from biomechanics literature.
* Enable semantic search over scientific papers.
* Compare multiple machine learning models for activation prediction.

---

## Features

### EMG Feature Extraction

The system extracts key EMG features:

* RMS (Root Mean Square)
* MAV (Mean Absolute Value)
* ZCR (Zero Crossing Rate)
* WL (Waveform Length)

### Activation Prediction

Predicts soleus muscle activation score using machine learning.

### AI Coach

Provides personalized recommendations based on activation levels.

### Research Assistant

Allows users to query biomechanics and EMG research papers using Retrieval-Augmented Generation (RAG).

### Model Comparison

Compares:

* Linear Regression
* Random Forest
* XGBoost

### FastAPI REST APIs

Provides a production-ready backend for integration with future hardware systems.

---

## Technology Stack

### Programming Language

* Python 3.14

### Machine Learning

* Scikit-Learn
* XGBoost

### Data Processing

* Pandas
* NumPy

### Backend

* FastAPI
* Uvicorn

### AI & RAG

* LangChain
* FAISS
* Sentence Transformers
* HuggingFace Embeddings

---

## Project Structure

```text
data/
├── raw/
├── processed/

models/
├── emg_activation_model.pkl
├── model_comparison.csv
├── model_comparison.png

src/
├── api/
├── features/
├── models/
├── preprocessing/
├── rag/

vectorstore/

README.md
```

---

## API Endpoints

### GET /

Returns API status.

Response:

```json
{
  "message": "SoleusAI Running"
}
```

---

### POST /predict

Predict soleus muscle activation.

Input:

```json
{
  "rms": 0.65,
  "mav": 0.52,
  "zcr": 95,
  "wl": 145
}
```

Output:

```json
{
  "activation_score": 66.33
}
```

---

### POST /coach

Provides training recommendations based on activation score.

Output Example:

```json
{
  "activation_score": 66,
  "recommendation": "Moderate Soleus Activation"
}
```

---

### POST /ask

Research paper question answering.

Input:

```json
{
  "question": "What factors influence soleus activation?"
}
```

Output:

```json
{
  "answer": "LLM-enhanced response generated from research papers."
}
```

---

### GET /model-info

Returns project and model information.

---

## Experimental Results

| Model             | R² Score |
| ----------------- | -------- |
| Linear Regression | 0.9801   |
| Random Forest     | 0.9785   |
| XGBoost           | 0.9740   |

Best Performing Model:

**Linear Regression (R² = 0.9801)**

---

## Applications

* Sports Science
* Rehabilitation Engineering
* Physiotherapy
* Biomechanics Research
* Wearable Health Monitoring

---

## Future Scope

* Real EMG Sensor Integration
* Live CSV Upload API
* Streamlit Dashboard
* Personalized AI Training Coach
* Real-Time Monitoring System
* Cloud Deployment

---

## Current Project Status

* EMG Feature Engineering Completed
* Machine Learning Pipeline Completed
* FastAPI Backend Completed
* RAG Knowledge Base Completed
* AI Coach Completed
* Model Comparison Completed
* GitHub Integration Completed

Project Status: **Working Prototype (MVP)**
