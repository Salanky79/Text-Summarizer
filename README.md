# End to End Text Summarizer Project

An end-to-end NLP project that fine-tunes the **PEGASUS** model (`google/pegasus-cnn_dailymail`) on the **SAMSum dataset** for automated dialogue summarization, built with FastAPI and modular design.

---

## 🛠️ Workflows

1. Update `config/config.yaml`
2. Update `params.yaml`
3. Update entity (`src/textSummarizer/entity`)
4. Update the Configuration Manager in `src/textSummarizer/config`
5. Update components (`src/textSummarizer/components`)
6. Update pipeline stages (`src/textSummarizer/pipeline`)
7. Update `main.py`
8. Update `app.py`

---

## 🚀 Model Training (Google Colab)

Due to local hardware limits, model fine-tuning was executed on **Google Colab (T4 GPU)**.

* **Dataset:** `knkarthick/samsum` (Hugging Face)
* **Model Checkpoint:** `google/pegasus-cnn_dailymail`
* **Training Metrics:** 1 Epoch, Training Loss ~1.63, Validation Loss 1.50
* **Colab Notebook:** Training script available at `research/04_model_trainer_colab.ipynb`

---

## 📦 Model Artifacts Setup

After downloading the trained model files from Google Drive, extract and place them inside the `artifacts` directory:

```text
artifacts/
└── model_trainer/
    └── pegasus-samsum-model/
        ├── config.json
        ├── generation_config.json
        ├── model-001.safetensors
        ├── model-002.safetensors
        ├── model.safetensors.index.json
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── spiece.model
```

---

## 🏃 How to Run the Application

### 1. Environment Setup
Create a virtual environment and install the required dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Install dependencies and local package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Launch FastAPI Server
Run the local prediction server:

```bash
python app.py
```

Access the Interactive API Documentation at: `http://localhost:8080/docs`

---

## 🎛️ Dynamic Summary Length Customization

The service supports dynamic summary lengths (`Short`, `Medium`, `Long`, `Detailed`) controlled via request parameters.

### How It Works

Instead of static generation, the pipeline adjusts PEGASUS beam search decoding hyper-parameters on the fly:

* `max_length`: Upper bound for generated sequence tokens.
* `min_length`: Forces the model to build context before stopping generation.
* `length_penalty`: Exponential penalty on sequence length (> 1.0 encourages longer summaries, < 1.0 favors shorter outputs).

| Summary Type | Max Length | Min Length | Length Penalty | Ideal Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Short** | 50 | 10 | 0.6 | TL;DR, quick key takeaway |
| **Medium** | 100 | 30 | 1.0 | Standard chat summary |
| **Long** | 200 | 80 | 1.5 | Detailed discussion notes |
| **Detailed** | 350 | 150 | 2.0 | Full transcript highlights |


## 📩 API Usage Example

**POST Request:** `http://localhost:8080/predict`

```json
{
  "text": "Hannah: Hey, have you seen the new project roadmap for next quarter? \nEric: Not yet, did Sarah upload it? \nHannah: Yeah, she pushed it about an hour ago. We need to review database migration.",
  "summary_type": "Short"
}
```

**Response (200 OK):**

```json
{
  "summary": "Sarah uploaded the new project roadmap for next quarter. Hannah and Eric need to review the database migration.",
  "summary_type": "Short"
}
```