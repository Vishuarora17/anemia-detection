# 🩸 Non-Invasive Anemia Detection Using Deep Learning on Ocular Images

## 📌 Overview
Anemia is a widespread global health condition that affects millions of people, particularly children and pregnant women. Conventional diagnosis relies on invasive blood tests, which are costly, time-consuming, and difficult to scale in low-resource settings.  

This project presents a **non-invasive deep learning–based approach** for anemia screening using images of the human eye. By analyzing ocular images, the system aims to provide an accessible and low-cost alternative to traditional diagnostic methods.

---

## 🎯 Objectives
- Develop deep learning models for non-invasive anemia detection  
- Compare multiple pretrained CNN architectures  
- Evaluate ensemble learning to improve clinical reliability  
- Address class imbalance commonly found in medical datasets  
- Prioritize high recall for anemic cases to minimize false negatives  

---

## 🧠 Models Implemented

### Individual Models
- **VGG16**
- **DenseNet121**
- **InceptionV3**

### Ensemble Models
- **VGG16 + DenseNet121 + InceptionV3**
- **DenseNet121 + InceptionV3**

Ensemble predictions are generated using probability averaging to leverage complementary feature representations from different architectures.

---

## 📊 Dataset
- **Input:** Ocular (eye / conjunctiva) images  
- **Classes:** Anemic, Non-Anemic  
- **Class Distribution:** ~3:1 (Non-Anemic : Anemic)  
- **Key Challenge:** Severe class imbalance  

The dataset reflects real-world screening scenarios, making recall-oriented evaluation essential.

---

## ⚙️ Preprocessing
- Image resizing to model-specific input dimensions  
- Pixel normalization  
- Data augmentation (rotation, flipping, zoom)  
- Stratified train–validation–test split  

---

## 📈 Evaluation Metrics
Models were evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Mean Squared Error (MSE)  
- Standard deviation across validation folds  

> **Recall for the anemic class** was treated as the most critical metric due to clinical safety considerations.

---

## 🧪 Results Summary

### Individual Models
- **InceptionV3** achieved the best standalone performance:
  - ~71% accuracy  
  - Lowest MSE  
  - Most stable results across folds  

- **VGG16** and **DenseNet121** showed limited recall for anemic cases, highlighting the impact of class imbalance.

### Ensemble Models
- **VGG16 + DenseNet121 + InceptionV3 (Best Model):**
  - **84% accuracy**
  - **0.75 recall for anemic class**
  - Best F1-scores across both classes  

- **DenseNet121 + InceptionV3:**
  - 81% accuracy  
  - Lower computational cost  
  - Reduced recall for anemic class compared to three-model ensemble  

### Key Insight
Ensemble learning significantly improves diagnostic reliability and mitigates weaknesses of individual models, making it more suitable for medical screening applications.

---

## 🏥 Medical Relevance
- False negatives (missing anemic patients) are more dangerous than false positives  
- High recall for anemic cases is critical for safe screening  
- Ensemble models provide a better balance between accuracy and clinical reliability  

---

## 🛠️ Tools & Technologies
- **Language:** Python  
- **Frameworks:** TensorFlow, Keras  
- **Libraries:** NumPy, OpenCV, Scikit-learn, Matplotlib  
- **Hardware:** GPU recommended for training  

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone <repository-url>
cd anemia-detection
