### 🌱 Plant Disease Detection System for Sustainable Agriculture  

## 🚀 Overview  
The **Plant Disease Detection System** leverages **Convolutional Neural Networks (CNNs)** to accurately detect and classify plant diseases from images of leaves. Designed for precision agriculture, this system aids farmers and agricultural experts in early disease detection and efficient crop management to reduce losses and enhance productivity.  

---

## 🎯 Project Aim  
To design and implement a CNN-based model capable of:  
- Identifying healthy and diseased leaves from leaf images.  
- Predicting specific disease types for crops such as apple, cherry, grape, and corn.  
- Supporting early diagnosis to improve crop management practices.  

---

## 📌 Learning Objectives  
1. **Dataset Acquisition and Preparation**  
   - Collect and preprocess a diverse dataset of healthy and diseased leaves.  
   - Ensure uniform image size, resolution, and quality.  

2. **Exploratory Data Analysis (EDA)**  
   - Analyze image characteristics like size, color distribution, and class balance.  

3. **Feature Extraction**  
   - Extract meaningful features (e.g., leaf texture, color patterns, vein structures) using CNN layers.  

4. **Model Development**  
   - Train and evaluate CNN architectures for image classification.  
   - Optimize hyperparameters for better performance.  

5. **Model Evaluation**  
   - Assess performance using metrics like accuracy, precision, recall, and F1-score.  

---

## 📊 Dataset  
- **Source:** [Kaggle - Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data?select=New+Plant+Diseases+Dataset%28Augmented%29)  
- **Details:**  
  - ~87,000 RGB images categorized into 38 classes (healthy and diseased).  
  - Split into 80/20 for training and validation, with a separate test set.  
  - Images stored in directories based on health and disease types.  

---

## 🛠️ Tools and Frameworks  
### 1. **Python**  
- Used for building the CNN model with frameworks like **TensorFlow** and **Keras**.  
- Supports image preprocessing and dataset augmentation.  

### 2. **Google Colab**  
- Cloud-based platform for developing and training deep learning models.  
- Access to powerful hardware (GPUs/TPUs) for efficient training.  
- Enables real-time collaboration and visualization.  

### 3. **Streamlit**  
- Framework for creating interactive data apps for disease detection.  

---

## ⚙️ Workflow  

### 1. **Data Preparation**  
- **Loading Dataset**: Organizing healthy and diseased leaf images for training/testing.  
- **Preprocessing**: Resizing images (e.g., 224x224 pixels) and normalizing pixel values.  

### 2. **Model Training**  
- Training a **CNN** model to detect and classify diseases.  
- Fine-tuning hyperparameters (learning rate, number of layers, kernel size).  

### 3. **Prediction Function**  
- Preprocesses input images for prediction.  
- Outputs the disease type or healthy condition.  

### 4. **Evaluation**  
- Tested on unseen data to ensure model generalization.  
- Metrics: Accuracy, Precision, Recall, and F1-score.  

---

## 🖼️ Features and Insights  
- **Classes**: 38 (healthy + diseased).  
- **Model Outputs**: Disease type or "Healthy".  
- **Interactive UI**: Built using Streamlit for user-friendly predictions.  

---

## 📈 Results  
- The model achieves high accuracy in classifying plant health conditions.  
- It demonstrates the potential of deep learning for precision agriculture.  

---

## 📚 Future Enhancements  
- Integrate real-time image capture via a mobile app.  
- Expand dataset to include more crops and disease types.  
- Deploy the model as a cloud-based API for scalability.  

---

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 📬 Contact
For any queries or contributions, feel free to reach out:
Email: connect.shreyaupadhyay@gmail.com
LinkedIn: Shreya Upadhyay (https://linkedin.com/in/shryay)

## 🛡️ All Rights Reserved
© 2024 Shreya Upadhyay. All rights reserved. This project and its contents are protected under copyright law.
