# Adversarial-Forensics-Investigation

**Case:** Professor Moriarty's "Digital Cloak" - Adversarial Attack Analysis  
**Dataset:** Caltech-101 (101 object categories)  
**Model Architecture:** MobileNetV2 (Transfer Learning)

---

## Executive Summary

This investigation exposed Professor Moriarty's "Digital Cloak" - a sophisticated adversarial attack technique that renders AI surveillance systems blind to criminal activities. Through systematic analysis of model vulnerabilities, implementation of explainability techniques, and development of defensive countermeasures, we have demonstrated the effectiveness of adversarial attacks and evaluated potential defense mechanisms.

### Key Findings

- **Model Performance : **Achieved 92.70% validation accuracy on clean data
- **Attack Effectiveness : **FGSM attacks reduced accuracy from 93.02% to 26.09% (66.93% degradation)
- **Defense Evaluation : **Adversarial training maintained clean performance while testing robustness
- **Forensic Analysis : **Visualizations for attack diagnosis

---

### Attack Methods Implemented

#### Fast Gradient Sign Method

**Implementation:**

```python
def fgsm_attack(self, model, data, target, epsilon=0.1):
    """FGSM"""

    data.requires_grad = True

    #Forward pass
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)

    #Backward pass
    model.zero_grad()
    loss.backward()

    #adversarial perturbation
    perturbation = epsilon * data.grad.sign()
    adversarial_data = data + perturbation

    #Clip to valid range
    adversarial_data = torch.clamp(adversarial_data, 0, 1)

    return adversarial_data, perturbation
```

**Methodology:**

- **Single-step attack** that computes the gradient of the loss with respect to the input
- **Perturbation magnitude** controlled by epsilon parameter
- **Sign function** ensures maximum gradient direction
- **Clipping** maintains pixel values in valid range [0,1]

### Attack Effectiveness Analysis

#### FGSM Attack Results

| Epsilon | Model Accuracy | Degradation |
| ------- | -------------- | ----------- |
| 0.01    | 58.10%         | 34.92%      |
| 0.05    | 47.51%         | 45.51%      |
| 0.1     | 40.37%         | 52.65%      |
| 0.15    | 32.23%         | 60.79%      |
| 0.2     | 26.09%         | 66.93%      |

**Key Observations:**

- **Linear degradation** in accuracy with increasing epsilon
- **Critical threshold** around epsilon=0.1 (50%+ degradation)
- **Diminishing returns** beyond epsilon=0.15
- **Stealth factor** - epsilon=0.01 causes significant damage (34.92% degradation)

#### Attack Visualization

The forensic analysis generated comprehensive visualizations showing:

- **Original vs. Adversarial Images:** Minimal visual differences
- **Saliency Maps:** Shift in model attention patterns
- **Grad-CAM:** Changes in feature activation regions
- **Perturbation Patterns:** Structured noise distribution

---

### Adversarial Training

#### Implementation

```python
def adversarial_training(self, model_name='mobilenet_v2', num_epochs=5):
    """Adversarial Training Defense"""

    model = self.models[model_name]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)

            #adversarial examples
            adv_data, _ = self.fgsm_attack(model, data.clone(), target, epsilon=0.1)

            # Combine clean and adversarial data
            combined_data = torch.cat([data, adv_data], dim=0)
            combined_target = torch.cat([target, target], dim=0)

            # Train on combined dataset
            optimizer.zero_grad()
            output = model(combined_data)
            loss = criterion(output, combined_target)
            loss.backward()
            optimizer.step()
```

#### Methodology

**Training Strategy:**

- **Data Augmentation :** 50% clean + 50% adversarial examples
- **Attack Type :** FGSM with epsilon=0.1
- **Learning Rate :** Reduced to 0.0001 for stable training
- **Epochs :** 5 epochs of adversarial training

**Key Parameters:**

- **Epsilon :** 0.1 (attack strength during training)
- **Batch Ratio :** 1:1 clean to adversarial
- **Optimizer :** Adam with reduced learning rate
- **Loss Function :** Cross-entropy on combined dataset

### The Showdown

#### Quantitative Results

| Metric               | Original Model | Defended Model | Improvement |
| -------------------- | -------------- | -------------- | ----------- |
| Clean Accuracy       | 93.02%         | 93.02%         | 0.00%       |
| Adversarial Accuracy | 65.00%         | 65.00%         | 0.00%       |
| Attack Degradation   | 28.02%         | 28.02%         | 0.00%       |

#### Analysis of Results

**Defense Performance:**

- **Clean Performance :** Maintained at 93.02%
- **Adversarial Robustness :** No improvement observed

**Potential Issues:**

1. **Limited Training :** 5 epochs may be insufficient for robust defense
2. **Model Capacity :** MobileNetV2 may have limited robustness potential
3. **Hyperparameter Tuning :** Learning rate and epsilon may need optimization

---

## Model Explainability Analysis

### Explainability Techniques Implemented

#### Saliency Maps

**Implementation:**

```python
def saliency_map(self, model, data, target):
    """Saliency Maps"""

    data = data.clone().detach().requires_grad_(True)

    output = model(data)
    loss = output[0, target]

    model.zero_grad()
    loss.backward()

    saliency_map = data.grad.abs().max(dim=1)[0]
    return saliency_map
```

**Methodology:**

- **Gradient computation** with respect to input pixels
- **Absolute values** to show importance magnitude
- **Channel-wise maximum** to create 2D saliency map
- **Visualization** using heatmap overlay

#### Grad-CAM

**Implementation:**

```python
def gradcam(self, model, data, target, layer_name='features'):
    """Grad CAM"""

    # Hook to get gradients and activations
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def save_activation(module, input, output):
        activations.append(output)

    # Register hooks on target layer
    target_layer = model.features[-1]
    handle_forward = target_layer.register_forward_hook(save_activation)
    handle_backward = target_layer.register_backward_hook(
        lambda module, grad_input, grad_output: save_gradient(grad_output[0]))

    # Forward and backward pass
    data = data.clone().detach().requires_grad_(True)
    output = model(data)
    loss = output[0, target]

    model.zero_grad()
    loss.backward()

    # Generate Grad-CAM
    gradients = gradients[0]
    activations = activations[0]
    weights = torch.mean(gradients, dim=[2, 3])

    gradcam = torch.zeros(activations.shape[2:], device=device)
    for i, w in enumerate(weights[0]):
        gradcam += w * activations[0, i, :, :]

    gradcam = torch.relu(gradcam)
    gradcam = gradcam / gradcam.max()

    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()

    return gradcam
```

**Methodology:**

- **Feature map extraction** from last convolutional layer
- **Gradient weighting** of feature maps
- **Global average pooling** of gradients
- **Weighted combination** of activation maps
- **ReLU activation** to focus on positive contributions

### Forensic Analysis Results

#### Visualization Components

Each forensic analysis example includes:

1. **Original Image :** Clean input with true/predicted labels
2. **Original Saliency Map :** Model attention on clean image
3. **Original Grad-CAM :** Feature activation on clean image
4. **Adversarial Perturbation :** Visual representation of attack noise
5. **Adversarial Image :** Perturbed input with new prediction
6. **Adversarial Saliency Map :** Model attention on attacked image
7. **Adversarial Grad-CAM :** Feature activation on attacked image
8. **Image Difference :** Visual comparison of changes

#### Key Observations

**Attack Patterns:**

- **Perturbation Distribution :** Structured noise patterns, not random
- **Attention Shifts :** Model focus moves from object features to background
- **Feature Confusion :** Grad-CAM shows activation in irrelevant regions
- **Stealth Effectiveness :** Minimal visual changes cause significant misclassification

**Defense Insights:**

- **Robustness Indicators :** Areas of consistent attention vs. vulnerable regions
- **Feature Importance :** Which visual elements are most critical for classification
- **Attack Susceptibility :** Regions where perturbations are most effective

---

### Model Architecture

**Base Model:** MobileNetV2

- **Transfer Learning :** Freeze all layers except the last convolutional layer
- **Classifier Replacement :** Custom head for 101 classes
- **Training Strategy :** Adam optimizer with learning rate scheduling

**Architecture Modifications:**

```python
model = models.mobilenet_v2(pretrained=True)

# Freeze all layers except the last one
for param in model.features[:-1].parameters():
    param.requires_grad = False
# Replace the classifier
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(model.classifier[1].in_features, self.num_classes)
)
```

### Dataset Processing

**Caltech-101 Dataset :**

- **Total Images :** 8,677 across 101 categories
- **Split :** 70% train, 15% validation, 15% test
- **Preprocessing :** Resize to 224×224, normalize with ImageNet statistics
- **Augmentation :** Standard normalization only (no augmentation for attack analysis)

### Training Configurati on

**Training Parameters :**

- **Batch Size :** 32 (reduced to 16 for adversarial training)
- **Learning Rate :** 0.001 (0.0001 for adversarial training)
- **Optimizer :** Adam with weight decay
- **Scheduler :** StepLR with gamma=0.1, step_size=7
- **Loss Function :** CrossEntropyLoss

---

**Training Results :**

- **Best Validation Accuracy :** 92.70% (Epoch 5)
- **Final Training Accuracy :** 99.79%
- **Overfitting :** Observed after epoch 5
- **Training Time :** ~1 hour for 10 epochs

**Test Results :**

- **Clean Test Accuracy :** 93.02%
- **Adversarial Test Accuracy :** 65.00% (FGSM, epsilon=0.1)
- **Performance Gap :** 28.02% degradation under attack

### Attack Effectiveness Analysis

**FGSM Attack Impact :**

- **Stealth Factor :** Epsilon=0.01 causes 34.92% accuracy drop
- **Critical Threshold :** Epsilon=0.1 achieves >50% degradation
- **Saturation Point :** Epsilon=0.2 provides 66.93% degradation
- **Practical Threat :** Even small perturbations are highly effective

**Attack Characteristics :**

- **Deterministic :** Same input always produces same perturbation
- **Scalable :** Works across different epsilon values

**Adversarial Training Results :**

- **Training Stability :** Model converged successfully
- **Clean Performance :** Maintained original accuracy
- **Robustness :** No improvement in adversarial accuracy
- **Computational Cost :** Significant training time increase

---
