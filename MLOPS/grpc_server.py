# grpc_server.py
import grpc
from concurrent import futures
import ml_service_pb2_grpc
import ml_service_pb2
import numpy as np
import joblib
import time

class MLServiceImpl(ml_service_pb2_grpc.MLServiceServicer):
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all available models"""
        self.models["v1.0"] = joblib.load("models/classifier_v1.pkl")
        self.models["v2.0"] = joblib.load("models/classifier_v2.pkl")
    
    def Predict(self, request, context):
        start_time = time.time()
        
        try:
            model = self.models.get(request.model_version, self.models["v1.0"])
            features = np.array(request.features).reshape(1, -1)
            
            prediction = model.predict(features)[0]
            confidence = max(model.predict_proba(features)[0])
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return ml_service_pb2.PredictResponse(
                prediction=prediction,
                confidence=confidence,
                model_version=request.model_version,
                processing_time_ms=processing_time
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Prediction failed: {str(e)}')
            return ml_service_pb2.PredictResponse()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(MLServiceImpl(), server)
    
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    
    print(f"Starting gRPC server on {listen_addr}")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()