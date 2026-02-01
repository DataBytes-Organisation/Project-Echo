import abc
import numpy as np
import tensorflow.lite as tflite
import onnxruntime as ort

class InferenceEngine(abc.ABC):
    @abc.abstractmethod
    def predict(self, input_data): pass

class TFLiteEngine(InferenceEngine):
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, input_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

class ONNXEngine(InferenceEngine):
    def __init__(self, model_path):
        # Load the ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_data):
        # ONNX Runtime expects inputs as a dictionary
        # input_data is already a numpy array from server.py
        result = self.session.run([self.output_name], {self.input_name: input_data})
        return result[0]

class ModelLoader:
    @staticmethod
    def get_engine(backend, path):
        if backend == "tflite": 
            return TFLiteEngine(path)
        elif backend == "onnx":
            return ONNXEngine(path)
        raise ValueError(f"Unknown backend: {backend}")