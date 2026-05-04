from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments
import json
import os


class T5SarcasmTranslator:
    PREFIX = "translate Slovenian sarcasm to literal Slovenian: "

    def __init__(self, model_name="t5-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    def load_data(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
        return data

    def create_dataset(self, data):
        dataset = Dataset.from_list(data)
        dataset = dataset.train_test_split(test_size=0.1)
        return dataset["train"], dataset["test"]

    def preprocess(self, example):
        prefix = self.PREFIX
        input_text = prefix + example["ironic"]
        target_text = example["literal"]

        model_inputs = self.tokenizer(
            input_text, max_length=64, truncation=True, padding="max_length"
        )
        labels = self.tokenizer(
            target_text, max_length=64, truncation=True, padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize(self, train_ds, val_ds):        
        train_ds = train_ds.map(self.preprocess, batched=False)
        val_ds = val_ds.map(self.preprocess, batched=False)
        return train_ds, val_ds

    def train(self, train_ds, val_ds, output_dir="./final_model"):
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=50,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def generate(self, text):
        prefix = self.PREFIX
        inputs = self.tokenizer(prefix + text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=64, num_beams=4)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def load_model(self, model_path, dataset=None):
        if not os.path.exists(model_path) and dataset is not None:
            print(f"Model not found at {model_path}. Training a new model...")
            self.create_and_train(dataset, model_path)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def create_and_train(self, dataset_path, output_dir="./final_model"):
        data = self.load_data(dataset_path)
        train_ds, val_ds = self.create_dataset(data)
        train_ds, val_ds = self.tokenize(train_ds, val_ds)
        self.train(train_ds, val_ds, output_dir)        