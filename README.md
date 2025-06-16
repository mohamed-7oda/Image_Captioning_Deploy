# 🧠 CaptionGo.AI – Image Captioning with Transformers

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Gradio](https://img.shields.io/badge/Gradio-UI-lightgrey)
![HuggingFace](https://img.shields.io/badge/Deployed-Huggingface-blueviolet)

## 🚀 About the Project
**Project Link** : https://huggingface.co/spaces/mohamed7oda/image-captioning?logs=container

**CaptionGo.AI** is an end-to-end deep learning project for automatic image captioning.  
Given an image, the model generates a natural language description using a Transformer decoder architecture.

The project is built with ❤️ as part of our Natural Language Processing course — where we tried multiple architectures, models, and techniques to push captioning accuracy further (within our compute limits 😅).

You can try the app live [**here**](#)  
📝 *Please use JPG images and avoid challenging, AI-generated, or blurry inputs — the model prefers clear, natural scenes!*

---

## 🧠 How It Works

1. **Dataset**: We used the [MS-COCO Dataset](https://cocodataset.org/#home) for training and evaluation.
2. **Preprocessing**:
   - Tokenization and cleaning of captions
   - Feature extraction from images using **ResNet152V2** (pre-trained on ImageNet)
3. **Model Architecture**:
   - Extracted visual features passed into a **Transformer decoder**
   - Trained to predict the next word in the caption sequence
4. **Frontend**:
   - Built using **Gradio** for a fast, interactive web UI
5. **Deployment**:
   - App deployed via **Hugging Face Spaces** for easy access and sharing

---

## ⚙️ Technologies Used

- Python 3.9
- TensorFlow / Keras
- Transformers (custom decoder)
- ResNet152V2 (CNN feature extractor)
- Gradio (for UI)
- Hugging Face Spaces (for deployment)

---

## 👨‍💻 Team

This project was developed by:

- Mohamed Ehab  
- Youssief Seliem  
- Amin Gamal  

With good vibes, too much trial-and-error, and a little GPU patience ⚡

---

## 🛠️ Run Locally

```bash
git clone https://github.com/yourusername/captiongo-ai.git
cd captiongo-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
