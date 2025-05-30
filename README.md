
# Multi-Disease Prediction Web App 🧠🩺

This project is a Flask-based web application that allows users to predict the risk or presence of various diseases using Machine Learning (ML) and Deep Learning (DL) models.

## 🔬 Supported Disease Models

- 🧠 Brain Tumor Detection (Image-based)
- ❤️ Cardiovascular Disease Prediction
- 🩸 Chronic Kidney Disease (CKD) Detection
- 💉 Diabetes Risk Prediction
- ⚖️ Obesity Risk Classification
- 🧬 Sickle Cell Anemia Detection
- 🩺 Iron Deficiency (Anemia) via Palm/Nail/Conjunctiva Image
- 🧪 Cancer Detection (Multi-class, Image-based)
- 🤒 Common Disease Symptom Checker (via NLP-based model)

---

## 🚀 Features

- 📦 Integrated multiple ML/DL models in a unified Flask app
- 📸 Image-based and form-based prediction support
- 💾 MySQL database integration for storing user submissions
- 🤖 AI Health Chatbot (Groq-powered) for symptom-based assistance
- 📁 Upload `.h5` and `.pkl` files handled via Git LFS

---

## 🧰 Tech Stack

| Component | Tech |
|----------|------|
| Backend  | Python, Flask |
| ML/DL    | scikit-learn, TensorFlow, Keras |
| DB       | MySQL |
| Frontend | HTML, CSS, Bootstrap |
| Deployment | GitHub (with Git LFS), Docker-ready |

---

## 🛠️ Setup Instructions

1. **Clone the repo**

   ```bash
   git clone https://github.com/deoprakash/multi_disease_prediction.git
   cd multi_disease_prediction
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**

   ```bash
   python app.py
   ```

4. **(Optional) Use Docker**

   ```bash
   docker compose up --build
   ```

---

## 📂 Project Structure

```
multi_disease_prediction/
│
├── app.py                      # Main Flask app
├── requirements.txt            # Python dependencies
├── templates/                  # HTML templates
├── static/                     # Uploaded images, CSS, JS
├── model/                      # Saved .h5/.pkl models
├── chatbot/                    # Groq chatbot integration
├── database/                   # MySQL connector & scripts
└── README.md                   # You are here
```

---

## 📌 Notes

- Git LFS is used to track large files like `.h5` (Keras models) and `.pkl` (pickle objects).  
  Run `git lfs install` before cloning.
- Set your `.env` for DB credentials and secret keys.

---

## 📄 License

MIT License. Free to use and modify.

---

## ✨ Author

**[Deo Prakash](https://www.linkedin.com/in/deo-prakash-152265225/)**  |  **[Arya Singh](https://www.linkedin.com/in/arya-singh-3558a5256/)**

Third-year B.Tech | AI/ML Engineer in progress 🚀
