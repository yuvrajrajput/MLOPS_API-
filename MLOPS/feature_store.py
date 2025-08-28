# feature_store.py
from abc import ABC, abstractmethod
import pandas as pd
import redis
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib

class FeatureStore(ABC):
    @abstractmethod
    def get_features(self, entity_id: str, feature_names: List[str]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def put_features(self, entity_id: str, features: Dict[str, Any], timestamp: datetime = None):
        pass

class RedisFeatureStore(FeatureStore):
    def __init__(self, redis_url: str, ttl: int = 3600):
        self.redis_client = redis.from_url(redis_url)
        self.ttl = ttl
    
    def _get_key(self, entity_id: str, feature_name: str) -> str:
        return f"feature:{entity_id}:{feature_name}"
    
    def get_features(self, entity_id: str, feature_names: List[str]) -> Dict[str, Any]:
        pipeline = self.redis_client.pipeline()
        keys = [self._get_key(entity_id, name) for name in feature_names]
        
        for key in keys:
            pipeline.get(key)
        
        results = pipeline.execute()
        
        features = {}
        for feature_name, result in zip(feature_names, results):
            if result:
                features[feature_name] = pickle.loads(result)
            else:
                features[feature_name] = None
                
        return features
    
    def put_features(self, entity_id: str, features: Dict[str, Any], timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        pipeline = self.redis_client.pipeline()
        
        for feature_name, value in features.items():
            key = self._get_key(entity_id, feature_name)
            feature_data = {
                'value': value,
                'timestamp': timestamp.isoformat(),
                'entity_id': entity_id
            }
            
            pipeline.setex(key, self.ttl, pickle.dumps(feature_data))
        
        pipeline.execute()

# Feature transformation pipeline
class FeaturePipeline:
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.transformations = {}
    
    def add_transformation(self, name: str, func: callable):
        """Add a feature transformation function"""
        self.transformations[name] = func
    
    def compute_features(self, entity_id: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute features from raw data"""
        features = {}
        
        for transform_name, transform_func in self.transformations.items():
            try:
                features[transform_name] = transform_func(raw_data)
            except Exception as e:
                print(f"Failed to compute feature {transform_name}: {e}")
                features[transform_name] = None
        
        # Store computed features
        self.feature_store.put_features(entity_id, features)
        
        return features

# Example transformations
def user_age_category(data: Dict[str, Any]) -> str:
    age = data.get('age', 0)
    if age < 18:
        return 'minor'
    elif age < 65:
        return 'adult'
    else:
        return 'senior'

def transaction_velocity(data: Dict[str, Any]) -> float:
    transactions = data.get('recent_transactions', [])
    if len(transactions) < 2:
        return 0.0
    
    time_diff = (max(transactions) - min(transactions)).total_seconds()
    return len(transactions) / max(time_diff, 1.0)

# Usage example
feature_store = RedisFeatureStore("redis://localhost:6379")
pipeline = FeaturePipeline(feature_store)
pipeline.add_transformation('age_category', user_age_category)
pipeline.add_transformation('transaction_velocity', transaction_velocity)