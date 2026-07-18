FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Code and Model
COPY server.py .
COPY inference_engine.py .

# ADD MODEL FILES HERE
COPY EFF2_QAT_Circle_clean.tflite model.tflite
COPY best_EFF2_QAT_Circle.onnx model.onnx

# Expose the standard TF Serving port
EXPOSE 8501

# Run the server
CMD ["python", "server.py"]