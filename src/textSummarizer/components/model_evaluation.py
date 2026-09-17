import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import evaluate

from textSummarizer.entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Chia dữ liệu thành các batch nhỏ."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        batch_size=16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        input_batches = list(
            self.generate_batch_sized_chunks(
                dataset["input_ids"],
                batch_size
            )
        )
        attention_batches = list(
            self.generate_batch_sized_chunks(
                dataset["attention_mask"],
                batch_size
            )
        )
        target_batches = list(
            self.generate_batch_sized_chunks(
                dataset["labels"],
                batch_size
            )
        )

        for input_batch, attention_batch, target_batch in tqdm(
            zip(input_batches, attention_batches, target_batches),
            total=len(input_batches)
        ):
            # TỰ ĐỘNG PAD BATCH VỀ CÙNG ĐỘ DÀI TRƯỚC KHÍ TẠO TENSOR
            batch_inputs = tokenizer.pad(
                {"input_ids": input_batch, "attention_mask": attention_batch},
                padding=True,
                return_tensors="pt"
            )

            input_ids = batch_inputs["input_ids"].to(device)
            attention_mask = batch_inputs["attention_mask"].to(device)

            # Dự đoán câu tóm tắt từ mô hình
            summaries = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                length_penalty=0.8,
                num_beams=8,
                max_length=128
            )

            # Giải mã kết quả do mô hình sinh ra
            decoded_summaries = [
                tokenizer.decode(
                    s,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                for s in summaries
            ]

            # Xử lý token -100 trong labels
            cleaned_targets = [
                [(t_id if t_id != -100 else tokenizer.pad_token_id) for t_id in target]
                for target in target_batch
            ]

            # Giải mã câu tóm tắt mẫu
            decoded_targets = [
                tokenizer.decode(
                    t,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                for t in cleaned_targets
            ]

            # Cập nhật kết quả vào bộ tính ROUGE
            metric.add_batch(
                predictions=decoded_summaries,
                references=decoded_targets
            )

        # Tính toán điểm ROUGE
        score = metric.compute()
        return score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path
        )

        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path
        ).to(device)

        model_pegasus.eval()

        # Nạp dữ liệu từ ổ đĩa
        dataset_samsum_pt = load_from_disk(
            self.config.data_path
        )

        rouge_names = [
            "rouge1",
            "rouge2",
            "rougeL",
            "rougeLsum"
        ]

        rouge_metric = evaluate.load("rouge")

        score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt["test"].select(range(10)),
            rouge_metric,
            model_pegasus,
            tokenizer,
            batch_size=2
        )

        rouge_dict = dict(
            (rn, score[rn])
            for rn in rouge_names
        )

        df = pd.DataFrame(
            rouge_dict,
            index=["pegasus"]
        )

        df.to_csv(
            self.config.metric_file_name,
            index=False
        )

        return df