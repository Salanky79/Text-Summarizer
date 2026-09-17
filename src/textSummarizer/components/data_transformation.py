import os
from textSummarizer.entity import DataTransformationConfig
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        # 1. Tối ưu: Dùng text_target trực tiếp, ngắn gọn và không bị Deprecation Warning
        input_encodings = self.tokenizer(
            example_batch['dialogue'], 
            max_length=1024, 
            truncation=True
        )
        target_encodings = self.tokenizer(
            text_target=example_batch['summary'], 
            max_length=128, 
            truncation=True
        )
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        
        # Lấy tên các cột chữ thô ('id', 'dialogue', 'summary')
        column_names = dataset_samsum['train'].column_names
        
        # 2. Tối ưu: Loại bỏ cột văn bản thô sau khi đã chuyển thành số
        dataset_samsum_pt = dataset_samsum.map(
            self.convert_examples_to_features, 
            batched=True,
            remove_columns=column_names
        )
        
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))