from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)

    def predict(self, text: str, summary_type: str = "Short") -> str:
        # Cấu hình tham số độ dài linh hoạt
        length_configs = {
            "Short":    {"max_length": 50,  "min_length": 10, "length_penalty": 0.6},
            "Medium":   {"max_length": 100, "min_length": 30, "length_penalty": 1.0},
            "Long":     {"max_length": 200, "min_length": 80, "length_penalty": 1.5},
            "Detailed": {"max_length": 350, "min_length": 150, "length_penalty": 2.0}
        }

        # Lấy tham số tương ứng (mặc định lấy "Short" nếu truyền sai)
        params = length_configs.get(summary_type, length_configs["Short"])

        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=8,
            max_length=params["max_length"],
            min_length=params["min_length"],
            length_penalty=params["length_penalty"]
        )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)