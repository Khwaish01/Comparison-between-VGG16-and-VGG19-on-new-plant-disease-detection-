# 🌿 Comparison between VGG16 and VGG19 on New Plant Disease Detection

## 📘 Overview
This project focuses on comparing the performance of **VGG16** and **VGG19**, two widely used convolutional neural network (CNN) architectures, for **plant disease detection**.  
Using transfer learning and fine-tuning, both models were trained to classify plant leaf images as **healthy** or **diseased**, based on the **New Plant Diseases Dataset** from Kaggle.

The objective is to determine which architecture provides better **accuracy**, **generalization**, and **efficiency** for plant disease classification.

---

## 📂 Dataset Details

- **Source:** [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  
- **Description:**  
  The dataset contains high-quality images of healthy and diseased plant leaves. It includes **38 distinct classes** covering different crops and disease types such as *Tomato Late Blight*, *Apple Scab*, *Corn Common Rust*, and *Healthy*.  
- **Image Type:** RGB images of individual leaves.  
- **Total Images:** ~87,000 samples.  
- **Split Used:**
  - **Training Set:** 80%  
  - **Validation Set:** 20%

### 🧹 Preprocessing
- **Image Size:** 224×224 pixels  
- **Normalization:** Pixel values scaled between 0 and 1  
- **Data Augmentation:**  
  - Rotation: **±45°**  
  - Width/Height Shift: 0.2  
  - Zoom Range: 0.2  
  - Horizontal & Vertical Flip: Enabled  
- **Batch Size:** **128**  
- **Shuffling:** Enabled for each epoch  

These augmentation techniques help prevent overfitting and improve the model’s ability to generalize across lighting, angles, and backgrounds.

---

## 🧠 Model Architectures

Both models used **ImageNet pre-trained weights** with the top classification layer replaced by a custom dense layer stack designed for multi-class classification.

### 🔹 VGG16
- 16 layers (13 convolutional + 3 fully connected)  
- Parameters: ~138 million  
- Easier to train, faster convergence, and more stable on moderately complex datasets  

### 🔹 VGG19
- 19 layers (16 convolutional + 3 fully connected)  
- Parameters: ~143 million  
- Deeper and heavier, with slightly slower training and a higher tendency to overfit  

---

## ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Categorical Crossentropy |
| Epochs | 10 |
| Batch Size | 128 |
| Framework | TensorFlow / Keras |
| Input Shape | 224 × 224 × 3 |

---

## 📊 Experimental Results

| Metric | **VGG16** | **VGG19** |
|--------|------------|------------|
| Training Accuracy | 90.78% | 88.03% |
| Validation Accuracy | **92.57%** | 91.76% |
| Training Loss | 0.3143 | 0.4127 |
| Validation Loss | **0.2567** | 0.2868 |
| Average Epoch Time | ~930s | ~940s |

### ✅ Observations
- **VGG16 achieved higher validation accuracy** and **lower validation loss** than VGG19.  
- **Training was more stable** for VGG16, showing smoother convergence.  
- **VGG19** tended to overfit slightly in later epochs, despite similar data augmentation settings.

---

## 🧩 Why VGG16 Performed Better

1. **Dataset Complexity vs. Model Depth**  
   The plant disease dataset features color and texture variations that are moderately complex.  
   **VGG16’s shallower architecture** is sufficient to extract relevant features, while **VGG19’s deeper structure** adds redundant layers that may lead to overfitting.

2. **Generalization Efficiency**  
   With aggressive data augmentation and a large batch size, **VGG16** generalized better to unseen validation data.  
   **VGG19**, having more parameters, required more epochs and potentially more data to reach its full potential.

3. **Optimization Stability**  
   VGG19’s deeper architecture makes it more susceptible to **vanishing gradients**, reducing training stability.  
   VGG16’s simpler architecture allows more consistent gradient updates and faster convergence.

4. **Computational Cost**  
   The higher parameter count in VGG19 increases both memory and computation time.  
   For this dataset, **VGG16 provided a better trade-off** between performance and efficiency.

---

## 📈 Training Summary

### VGG16 (10 Epochs)
```
Epoch 10/10
accuracy: 0.9078 - loss: 0.3143 - val_accuracy: 0.9257 - val_loss: 0.2567
```

### VGG19 (10 Epochs)
```
Epoch 10/10
accuracy: 0.8803 - loss: 0.4127 - val_accuracy: 0.9176 - val_loss: 0.2868
```

---

## 📚 References
- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet).*  
- [Keras VGG16 & VGG19 Documentation](https://keras.io/api/applications/vgg/)  
- [New Plant Diseases Dataset — Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

## 🏁 Conclusion
This study demonstrates that:
> ✅ **VGG16** achieved superior validation accuracy (**92.57%**) and lower loss (**0.2567**) than **VGG19** on the *New Plant Diseases Dataset*.

Although VGG19 is deeper, the **simpler VGG16 architecture** proved more effective and computationally efficient for this task.  
It captured the necessary spatial and texture-based features of plant diseases without overfitting, making it a better choice for **real-world agricultural image classification systems**.
