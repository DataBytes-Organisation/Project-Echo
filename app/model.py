# app/model.py
import random, time

class DummyModel:
    def __init__(self):
        self.name = "dummy-audio-classifier"
        self.labels = ["bird", "frog", "insect", "mammal", "rain", "wind", "unknown"]

    def predict(self, audio_bytes: bytes) -> str:
        # 模拟推理耗时与结果（后续替换为真实模型推理）
        time.sleep(0.1)
        random.seed(len(audio_bytes))
        return random.choice(self.labels)
