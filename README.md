MLOps API
A production-ready MLOps API for serving machine learning predictions with automated data pipelines and experiment tracking.

📌 Overview
This project provides a REST API to serve ML model predictions, automating the full pipeline from raw data ingestion to model inference. Experiment tracking and model versioning are handled via MLflow, making it easy to compare runs and deploy the best-performing model.

✨ Features

⚡ FastAPI — high-performance REST endpoints for real-time predictions
🔁 Automated Data Pipeline — handles preprocessing and feature engineering automatically
🧪 MLflow Tracking — logs experiments, parameters, metrics, and model versions
🤖 Scikit-learn Models — trained, versioned, and served via API
📊 Prediction Endpoint — send raw input, get back predictions instantly


🛠️ Tech Stack
ToolPurposeFastAPIREST API frameworkScikit-learnML model training & inferenceMLflowExperiment tracking & model registryUvicornASGI serverPandas / NumPyData pipeline & preprocessing

📁 Project Structure
mlops-api/
│
├── app/
│   ├── main.py          # FastAPI app & routes
│   ├── model.py         # Model loading & prediction logic
│   └── pipeline.py      # Data preprocessing pipeline
│
├── mlflow/
│   └── experiments/     # MLflow tracked runs
│
├── models/
│   └── model.pkl        # Trained scikit-learn model
│
├── requirements.txt
└── README.md

⚙️ Getting Started
1. Clone the repository
bashgit clone https://github.com/yuvrajrajput/mlops-api.git
cd mlops-api
2. Install dependencies
bashpip install -r requirements.txt
3. Start the API
bashuvicorn app.main:app --reload
4. Launch MLflow UI
bashmlflow ui
Open http://localhost:5000 to view experiments.

📡 API Usage
Predict Endpoint
POST /predict
json// Request
{
  "feature_1": 5.1,
  "feature_2": 3.5,
  "feature_3": 1.4
}

// Response
{
  "prediction": 1,
  "confidence": 0.94
}
Health Check
GET /health
json{ "status": "ok" }

📈 MLflow Experiment Tracking
All training runs are tracked with:

Model parameters (e.g. n_estimators, max_depth)
Metrics (accuracy, F1, AUC)
Artifacts (trained model, confusion matrix)

To register the best model:
bashmlflow models serve -m "models:/BestModel/Production" --port 1234
