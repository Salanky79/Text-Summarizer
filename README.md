# End to End Text Summarizer Project

An end-to-end NLP project that fine-tunes the **PEGASUS** model (`google/pegasus-cnn_dailymail`) on the **samsung dataset** for automated dialogue summarization.

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