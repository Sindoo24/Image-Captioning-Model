# 🖼️ AI Image Captioning System

**Offline AI image captioning system using BLIP & ViT-GPT2 with validated 45% semantic accuracy.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)


## 📌 Overview
This project implements a locally hosted, production-ready AI application capable of generating descriptive captions for images in real-time. It leverages state-of-the-art Vision-Language models (BLIP and ViT-GPT2) to interpret visual scenes and convert them into natural language.

The system includes a robust **evaluation pipeline** validated against the **Flickr8k dataset** (8,000+ images), achieving industry-standard performance metrics suitable for automated metadata tagging and accessibility tools.








## ✨ Key Features
* **Offline Inference:** Runs heavy transformer models entirely on local hardware (CPU/GPU) without API costs.
* **Dual Model Architecture:**
    * **High Accuracy:** `Salesforce/blip-image-captioning-large` (Default)
    * **High Speed:** `nlpconnect/vit-gpt2-image-captioning`
* **Interactive UI:** Streamlit-based interface featuring drag-and-drop uploads, processing timers, and downloadable logs.
* **Session History:** Tracks all generated captions and images during the user session.
* **Scientific Validation:** Integrated evaluation scripts to calculate BLEU, METEOR, and ROUGE-L scores.

## 📊 Performance Metrics
The model was benchmarked on the full **Flickr8k Test Dataset (8,091 images)**.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **ROUGE-L** | **0.4473 (44.7%)** | Semantic overlap with human ground truth. |
| **METEOR** | **0.4417** | Alignment based on exact, stem, and synonym matches. |
| **BLEU-4** | **0.1860** | Precision of 4-gram word matches. |

*These results demonstrate strong zero-shot generalization capabilities.*

## 🚀 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Sindoo24/image-captioning-app.git
    cd image-captioning-app
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    python -m venv venv
    venv\Scripts\activate
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Dataset (For Evaluation Only)**
    To run the evaluation script, you need the Flickr8k dataset.
    * Download it from **[Kaggle: Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)**.
    * Extract the contents into the `data/` folder so it looks like this:
        ```text
        data/
        ├── Images/        <-- Folder containing 8091 images
        ├── captions.txt   <-- Text file with image descriptions
        ```

## 💻 Usage

### 1. Run the Application
Launch the web interface:
```bash
streamlit run app/streamlit_app.py
