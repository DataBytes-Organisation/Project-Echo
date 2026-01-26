[<- Back to Main Guide](model_integration_guide.md)

---

## Making Predictions More Distinct: Metric Learning

To improve classification robustness, we employ **Metric Learning**. This technique trains the model to generate feature vectors (embeddings) that are more discriminative than those produced by standard classification training.

### Standard Classification Weakness

A standard classifier learns to draw a decision boundary between classes. This is effective, but it may not result in a robust feature representation; the features may only be "just good enough" to separate the training data.

### Metric Learning Approach

Metric Learning refines the feature space with two objectives:
1.  Minimise the distance between embeddings from the **same class** (intra-class variance).
2.  Maximise the distance between embeddings from **different classes** (inter-class variance).

This is achieved using two components that replace the standard final classification layer.

1.  **The `CosineLinear` Layer**
    -   This layer calculates the **cosine similarity** between an input embedding and a set of weight vectors representing each class.
    -   Unlike a standard `nn.Linear` layer's dot product, this focuses on the **angle** (or direction), not the magnitude. The model's prediction is based on how closely aligned the audio's embedding is with each class's idealised feature vector.

2.  **`ArcFace` / `CircleLoss` (The Angular Margin)**
    -   These are loss-modifying functions applied during training to enforce larger separations between classes in the angular space.
    -   **Analogy**: Consider assigned seating in a theatre. A standard process checks your ticket and lets you sit anywhere in your assigned row. This approach checks your ticket, then ensures you are seated squarely in the centre of your assigned seat, creating a clear buffer to the next person.
    -   It achieves this by adding an **angular margin penalty** to the loss calculation for correct predictions. This makes the training task harder, forcing the model to produce embeddings that are more tightly clustered within their correct class and pushed further from incorrect classes. `CircleLoss` is a more recent and often more effective variant of this concept.

### Implementation Details

-   **Activation**: In `config.yaml`, set `training.use_arcface: "circle"` or `"arcface"`.
-   **Model Build**: The model factory (`model/__init__.py`) replaces the architecture's final `nn.Linear` layer with a `CosineLinear` layer.
-   **Training Loop**: The `Trainer` applies the `CircleLoss` or `ArcMarginProduct` function to the model's outputs before calculating the final training loss, thereby incorporating the margin penalty.

---
[<- Back to Main Guide](model_integration_guide.md)
