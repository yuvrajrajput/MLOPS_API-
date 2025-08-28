# src/train_model.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from datetime import datetime

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_and_prepare_data(data_config: dict) -> tuple:
    """Load and prepare training data"""
    # Load data (example with CSV)
    data = pd.read_csv(data_config['path'])
    
    # Feature engineering
    X = data[data_config['features']]
    y = data[data_config['target']]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, model_config: dict):
    """Train the ML model"""
    model = RandomForestClassifier(**model_config['hyperparameters'])
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
    
    # Fit the model
    model.fit(X_train, y_train)
    
    return model, cv_scores

def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1_score': f1_score(y_test, y_pred, average='macro')
    }
    
    return metrics

def register_model(model, metrics: dict, cv_scores, config: dict):
    """Register model with MLflow"""
    
    # Set MLflow experiment
    mlflow.set_experiment(config['experiment_name'])
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(config['model']['hyperparameters'])
        mlflow.log_param("data_version", config['data']['version'])
        
        # Log metrics
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=config['model']['name']
        )
        
        # Log additional artifacts
        mlflow.log_dict(config, "config.yaml")
        
        # Add tags
        mlflow.set_tags({
            "training_date": datetime.now().isoformat(),
            "environment": config.get('environment', 'development'),
            "model_type": "RandomForestClassifier"
        })
        
        return run.info.run_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--validate-threshold', type=float, default=0.8, 
                       help='Minimum F1 score threshold for model acceptance')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(config['data'])
    
    # Train model
    model, cv_scores = train_model(X_train, y_train, config['model'])
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    print(f"Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"Cross-validation F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Validate performance threshold
    if metrics['f1_score'] < args.validate_threshold:
        raise ValueError(f"Model F1 score {metrics['f1_score']:.4f} below threshold {args.validate_threshold}")
    
    # Register model with MLflow
    run_id = register_model(model, metrics, cv_scores, config)
    print(f"Model registered with run ID: {run_id}")
    
    # Save model locally for deployment
    os.makedirs('./models/latest', exist_ok=True)
    joblib.dump(model, './models/latest/model.pkl')
    
    with open('./models/latest/metrics.yaml', 'w') as f:
        yaml.dump(metrics, f)

if __name__ == "__main__":
    main()