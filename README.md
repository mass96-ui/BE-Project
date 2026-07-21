# BE-Project
Principal
# SoleusAI

AI-Powered Soleus Muscle Activation Optimization System

## Features

- EMG Feature Extraction (RMS, MAV, ZCR, WL)
- Soleus Activation Prediction using Random Forest
- AI-based Resistance Recommendation
- Research Paper Knowledge Base
- Retrieval-Augmented Research Assistant
- FastAPI REST APIs

## Project Structure

```text
data/
├── raw/
├── processed/

models/
├── activation_model.pkl
├── emg_activation_model.pkl

src/
├── api/
├── features/
├── models/
├── preprocessing/
├── rag/
```

## Tech Stack

- Python 3.14
- Scikit-Learn
- Pandas
- FastAPI
- LangChain
- FAISS
- Sentence Transformers
- HuggingFace Embeddings

## API Endpoints

### GET /

Returns API status.

### POST /predict

Predicts soleus muscle activation score.

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

### POST /ask

Research paper question answering.

Input:

```json
{
  "question": "What is soleus muscle activation?"
}
```

## Current Status

- EMG AI Pipeline Completed
- RAG Knowledge Base Completed
- FastAPI Backend Completed
- GitHub Integration Completed

## Future Work

- Real EMG Sensor Integration
- Personalized AI Coach
- LLM-based Answer Generation
<<<<<<< HEAD
- Real-time Soleus Monitoring
=======
- Real-time Soleus Monitoring
>>>>>>> e72b127fedf55f5959e6d89500c202f87e20fd2c
