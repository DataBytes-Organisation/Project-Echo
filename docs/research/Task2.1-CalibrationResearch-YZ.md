# Task 2.1 – Calibration Research (YZ)

## 1. Introduction

Project Echo focuses on multi-class audio classification to identify different animal species based on their acoustic signals. While modern deep learning models such as CNNs and transformer-based architectures can achieve high classification accuracy, their predicted probabilities are often poorly calibrated.

In practice, this means that the confidence scores produced by the model do not always reflect the true likelihood of correctness. For example, a model may assign a prediction probability of 90%, while the actual accuracy for such predictions is significantly lower. This issue reduces the reliability of the system, especially when confidence scores are used for decision-making or filtering predictions.

Therefore, post-hoc calibration methods are required to improve the reliability of predicted probabilities without retraining the model.

In the context of bioacoustic classification, as highlighted in previous literature, deep learning models such as convolutional neural networks (CNNs) and transformer-based architectures are widely used for extracting meaningful features from audio signals and achieving high classification accuracy. However, these models are primarily optimized for accuracy rather than probability reliability.

As a result, even well-performing models may produce poorly calibrated confidence scores. This issue becomes more critical in applications such as Project Echo, where predictions may be used to support ecological monitoring, species tracking, and decision-making processes. Therefore, improving probability calibration is essential to ensure the trustworthiness of the system.

---

## 2. Calibration Methods

### 2.1 Temperature Scaling

Temperature scaling is a widely used post-hoc calibration technique for deep learning models. It introduces a single scalar parameter (T) to adjust the logits before applying the softmax function.

This method preserves the relative ordering of class probabilities while scaling the confidence levels.

**Advantages:**
- Simple and computationally efficient
- Works well with deep neural networks
- Maintains class ranking
- Naturally supports multi-class classification

**Disadvantages:**
- Limited flexibility (only one parameter)

---

### 2.2 Platt Scaling

Platt scaling uses a logistic regression model to map predicted scores to calibrated probabilities.

It was originally designed for binary classification tasks and is commonly applied using a one-vs-rest approach in multi-class settings.

**Advantages:**
- Well-established method
- Easy to implement

**Disadvantages:**
- Not inherently suitable for multi-class problems
- Requires additional adaptation
- Less commonly used for deep learning models

---

### 2.3 Isotonic Regression

Isotonic regression is a non-parametric calibration method that fits a monotonic function to map predicted probabilities to calibrated values.

Unlike parametric methods, it can capture more complex relationships between predicted and true probabilities.

**Advantages:**
- Highly flexible
- Can model complex calibration patterns

**Disadvantages:**
- Prone to overfitting with limited data
- Requires more calibration data
- More complex to integrate into production systems

---

## 3. Comparison of Methods

| Method                | Multi-class Support | Complexity | Overfitting Risk | Practical Use |
|----------------------|-------------------|------------|------------------|--------------|
| Temperature Scaling  | Yes               | Low        | Low              | High         |
| Platt Scaling        | Limited           | Low        | Medium           | Medium       |
| Isotonic Regression  | Yes               | High       | High             | Low          |

---

## 4. Evaluation Considerations

To assess the effectiveness of calibration methods, it is important to evaluate how well predicted probabilities align with actual outcomes.

One commonly used metric is Expected Calibration Error (ECE), which measures the difference between predicted confidence and observed accuracy across different probability intervals. A lower ECE indicates better calibration.

Another useful tool is the reliability diagram, which visually represents the relationship between predicted probabilities and actual accuracy. A well-calibrated model should produce a curve close to the diagonal line.

In the context of Project Echo, these evaluation methods are essential to ensure that calibration improves the reliability of predictions without negatively affecting classification accuracy.

---

## 5. Recommended Approach

For Project Echo, Temperature Scaling is the most suitable calibration method.

This recommendation is based on the following considerations:

- The system uses deep learning models for multi-class audio classification
- Temperature scaling is specifically designed for neural network outputs
- It does not change the ranking of predictions, which is important for classification tasks
- It is easy to integrate into the existing inference pipeline (e.g., echo_engine.py)
- It provides stable performance without requiring large calibration datasets

In contrast, isotonic regression may introduce overfitting, while Platt scaling is less suitable for multi-class scenarios.

---

## 6. Conclusion

This study evaluated three post-hoc calibration methods for improving probability reliability in Project Echo. Among them, Temperature Scaling provides the best balance between simplicity, stability, and effectiveness for multi-class deep learning models.

It is therefore recommended as the calibration approach for future implementation in the Project Echo inference pipeline.