# Task 2.1 – Calibration Research (YZ) – Sprint 2 Update

## 1. Introduction

Project Echo uses deep learning models for multi-class audio classification. In Sprint 1, several calibration methods were researched and compared, including Temperature Scaling, Platt Scaling, and Isotonic Regression. Based on the comparison, Temperature Scaling was recommended because it is simple, efficient, and suitable for deep learning models.

This Sprint 2 update extends the previous calibration research by investigating how calibration could be integrated into the Project Echo inference and benchmarking pipeline.

The purpose of this Sprint 2 update was to:

- Investigate where calibration could be applied in the existing pipeline
- Understand how confidence scores are generated
- Review methods for evaluating calibration quality
- Consider the efficiency of calibration for edge deployment

---

## 2. Calibration Integration Investigation

Before implementing calibration, it was necessary to understand how prediction probabilities are generated within the Project Echo pipeline. The current system uses deep learning classification models that output logits, which are then converted into probabilities using the softmax function.

Although these probabilities appear confident, previous research has shown that deep neural networks are often overconfident. This means the predicted confidence values may not accurately represent the true likelihood of correctness.

For example, a prediction with 95% confidence may not actually be correct 95% of the time. In practical applications such as wildlife monitoring and species recognition, unreliable confidence scores may reduce trust in the system.

Because of this, calibration becomes important for improving probability reliability.

Temperature Scaling was selected as the preferred calibration method because it works particularly well with neural network outputs while keeping the implementation simple.

The calibration process can be added after inference without changing the original training process.

The proposed process is:

1. The model produces logits during inference
2. Logits are divided by a temperature parameter (T)
3. Softmax is applied to generate calibrated probabilities
4. The calibrated confidence scores are used for final predictions

The temperature parameter is usually learned using a validation dataset. A larger temperature value reduces overconfidence by softening the probability distribution.

One important advantage is that Temperature Scaling does not change the predicted class labels. Instead, it only adjusts the confidence values, making the predictions more reliable while preserving model accuracy.

This approach allows probability adjustment without retraining the entire model.

---

## 3. Evaluation Preparation

Several evaluation methods were investigated to measure calibration quality.

### 3.1 Expected Calibration Error (ECE)

ECE measures the difference between model confidence and actual prediction accuracy.

A lower ECE value indicates that the confidence scores are more reliable.

ECE is useful because:
- It is commonly used in calibration research
- It is simple to interpret
- It works well for multi-class classification

### 3.2 Reliability Diagram

A reliability diagram visually compares prediction confidence with actual accuracy.

A well-calibrated model should produce results close to the diagonal line.

This visualization helps identify whether the model is overconfident or underconfident.

---

## 4. Pipeline Considerations

During this sprint, the existing Project Echo benchmarking and experimentation pipeline was reviewed in more detail.

The current framework already supports model training, benchmarking, and evaluation for different architectures. Therefore, calibration should ideally be integrated in a way that does not interrupt the existing workflow.

Several possible integration locations were considered:

### 4.1 Inference Pipeline

Calibration can be added directly after model inference.

In this approach:
- The model first generates logits
- Temperature Scaling is applied
- Calibrated probabilities are returned to the user

This is the most direct and practical approach because it improves reliability during real-time prediction.

### 4.2 Benchmark Evaluation Stage

Calibration can also be included during benchmarking.

This would allow comparison between:
- Original probabilities
- Calibrated probabilities
- ECE before calibration
- ECE after calibration

This approach is useful for measuring how much improvement calibration provides.

### 4.3 Confidence Score Output Stage

Another possible integration point is the final confidence reporting stage.

This is important because users may rely on confidence values when interpreting predictions. More reliable confidence scores improve trustworthiness and decision-making.

The calibration process should remain lightweight so it can still support edge-optimized deployment.

Temperature Scaling is suitable for this because it only introduces one additional parameter and does not significantly increase computational cost.

Compared with more complex methods such as isotonic regression, Temperature Scaling is easier to maintain and more suitable for production environments.

---

## 5. Extended Investigation

The current Project Echo pipeline already supports:

- Model training
- Benchmarking
- Inference
- Confidence score generation

Because of this, Temperature Scaling can potentially be added without major structural changes.

Compared with retraining-based approaches, post-hoc calibration is more practical because:

| Approach              | Retraining Required | Complexity | Deployment Impact |
|-----------------------|---------------------|------------|-------------------|
| Temperature Scaling   | No                  | Low        | Low               |
| Full Model Retraining | Yes                 | High       | High              |

Temperature Scaling is therefore more suitable for Project Echo because it maintains efficiency while improving probability reliability.

---

## 6. Progress Summary

This sprint achieved the following progress:

- Continued the calibration task from Sprint 1
- Investigated how Temperature Scaling could be integrated into Project Echo
- Reviewed confidence score generation in the inference pipeline
- Investigated ECE and reliability diagrams for evaluation
- Considered deployment efficiency for edge environments

---

## 7. Future Work and Implementation Direction

The next step is to implement Temperature Scaling within the Project Echo pipeline and test calibration performance using model outputs.

Future work includes:
- Running experiments on prediction outputs
- Comparing confidence scores before and after calibration
- Measuring ECE improvements
- Generating reliability diagrams
- Evaluating whether calibration affects classification accuracy

---

## 8. Conclusion

This Sprint 2 update expanded the original calibration research by further investigating implementation considerations and evaluation methods for Project Echo.

Temperature Scaling remains the recommended calibration approach because it is simple, computationally efficient, and suitable for multi-class deep learning models.

The additional investigation completed during Sprint 2 provides a stronger foundation for future implementation and testing within the Project Echo inference pipeline.

