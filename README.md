

# 🌿 PlantDiseaseAI – Plant Disease Detection Using Deep Learning

An AI-powered system that detects **tomato plant diseases** from leaf images using **deep learning and transfer learning**.
The project includes **dataset preparation, data augmentation, MobileNetV2 model training, fine-tuning**, and a **Streamlit web application** for real-time image classification.

---

## 🚀 Features

* 🌱 Classifies leaf images as **Healthy** or **Diseased**
* 🧠 **Transfer Learning** using MobileNetV2
* 🧪 **Two-phase training:**

  * Train top layers
  * Fine-tune MobileNetV2
* ⚖ Uses **class weights** to handle dataset imbalance
* 🖼 Automatic image preprocessing
* 📈 Displays predicted class + confidence score
* 🌐 Simple and interactive **Streamlit Web App**

---

## 📁 Project Structure

```
PlantDiseaseAI/
│── app.py                 # Streamlit web application
│── train_model.py         # Model training script
│── utils.py               # Prediction + preprocessing utilities
│── balance_healthy.py     # Data augmentation for class balancing
│── organize_dataset.py    # Dataset creation (Healthy/Diseased)
│── model/
│   └── plant_disease_model.keras
│── dataset/
│   ├── train/
│   │   ├── Healthy/
│   │   └── Diseased/
│   └── valid/
│       ├── Healthy/
│       └── Diseased/
└── README.md
```

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/anushkajagdale/PlantDiseaseAI.git
cd PlantDiseaseAI
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

*(If you need a ready `requirements.txt`, tell me — I will generate one.)*

---

## 🧠 Model Training

The complete model training pipeline is implemented in **`train_model.py`** and includes:

### ✔ Model Architecture & Techniques

* MobileNetV2 (pre-trained on ImageNet)
* GlobalAveragePooling
* Dense layers with softmax
* Dropout regularization
* Image augmentation
* Class weights for imbalance
* Two-phase training (feature extraction + fine-tuning)

### ▶ Train the model

```bash
python train_model.py
```

This saves the trained model to:

```
model/plant_disease_model.keras
```

---

## 🌐 Running the Web App

Start the Streamlit application:

```bash
streamlit run app.py
```

Upload any tomato leaf image → the model predicts:

* **Healthy**, or
* **Diseased**

with a **confidence score**.

---

## 🔍 Prediction Pipeline

The prediction logic (in `utils.py`) handles:

* Loading the trained MobileNetV2 model
* Resizing input image to 224×224
* Scaling pixel values
* Predicting class index
* Mapping index → `["Healthy", "Diseased"]`

---

## 📊 Dataset Preparation

### ✔ Step 1: Organize Dataset

Creates folders:

```
dataset/train/Healthy
dataset/train/Diseased
dataset/valid/Healthy
dataset/valid/Diseased
```

Run:

```bash
python organize_dataset.py
```

### ✔ Step 2: Balance Dataset

Augments Healthy images until both classes match:

```bash
python balance_healthy.py
```

## 💡 Future Improvements

* Deploy app on AWS / Render / Heroku
* Detect multiple diseases (Early Blight, Late Blight, etc.)
* Add Grad-CAM heatmaps for explainability
* Create a mobile app version
* Improve UI with Streamlit components

---

## 🏆 Author

**Anushka Sopan Jagdale**
B.Tech (IT), Cummins College of Engineering, Pune
GitHub: [https://github.com/anushkajagdale](https://github.com/anushkajagdale)

