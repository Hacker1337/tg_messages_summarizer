import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import pipeline
from datasets import Dataset, DatasetDict, load_dataset
import evaluate
import wandb
import numpy as np

def make_dataset(dataframe):
    data = dataframe[["dialogue", "summary"]]
    dataset = Dataset.from_pandas(data)
    return dataset

def preprocess_function(examples):
    prefix = ""
    inputs = [prefix + doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result_rouge = rouge.compute(predictions=decoded_preds,
                        references=decoded_labels,
                        tokenizer=tokenizer.tokenize)

    bleu_results = bleu.compute(predictions=decoded_preds, references=decoded_labels, tokenizer=tokenizer.tokenize)
    result = {
        **{"rouge_" + k: v for k, v in result_rouge.items()},
        **{"bleu_" + k: v for k, v in bleu_results.items()},
    }
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) if isinstance(v, float) else v
            for k, v in result.items()}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune an ML model for summarization.")
    parser.add_argument("--train_dataset_size", type=int, default=13000, help="Size of the training dataset.")
    parser.add_argument("--validation_dataset_size", type=int, default=100, help="Size of the validation dataset.")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size for training and evaluation.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--checkpoint", type=str, default="IlyaGusev/rut5_base_headline_gen_telegram", help="Checkpoint to load from HuggingFace Hub.")
    parser.add_argument("--data_path", type=str, default=None, help="path to dataset files")
    parser.add_argument("--dataset_cloud_name", type=str, default="Hacker1337/ru_dialogsum", help="name of the dataset in HuggingFace Hub.")
    args = parser.parse_args()

    if args.data_path is not None:
        data_val = pd.read_csv(os.path.join(args.data_path, "output_validation.csv"))
        data_train = pd.read_csv(os.path.join(args.data_path, "output_train.csv"))

        dataset_train = make_dataset(data_train)
        dataset_val = make_dataset(data_val)
        dataset_dict = DatasetDict({
        "train": dataset_train,
        "validation": dataset_val,
        })
    else:
        dataset_dict = load_dataset(args.dataset_cloud_name) 

    # Model and tokenizer loading
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)

    n_samples = args.validation_dataset_size
    validation_data = dataset_dict["validation"].select(range(n_samples))

    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device="cuda")

    table_before_fine_tuning = wandb.Table(columns=["Input Text", "Target Summary", "Generated Summary"])

    for example in validation_data:
        input_text = example["dialogue"]
        target_summary = example["summary"]
        generated_summary = summarizer(input_text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        table_before_fine_tuning.add_data(input_text, target_summary, generated_summary[0]["summary_text"])

    wandb.log({"summarization_before_fine_tuning": table_before_fine_tuning})

    # Preprocess
    tokenized_dataset = dataset_dict.map(preprocess_function, batched=True)

    # Evaluate
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.checkpoint)

    training_args = Seq2SeqTrainingArguments(
        output_dir="second_llm_finetune_rubert",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=100,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        generation_max_length=100,
        report_to="wandb",  # enable logging to W&B
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"].select(range(args.train_dataset_size)),
        eval_dataset=tokenized_dataset["validation"].select(range(args.validation_dataset_size)),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # trainer.push_to_hub()

    n_samples = args.validation_dataset_size
    validation_data = dataset_dict["validation"].select(range(n_samples))

    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device="cuda")

    table_after_fine_tuning = wandb.Table(columns=["Input Text", "Target Summary", "Generated Summary"])

    for example in validation_data:
        input_text = example["dialogue"]
        target_summary = example["summary"]
        generated_summary = summarizer(input_text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        table_after_fine_tuning.add_data(input_text, target_summary, generated_summary[0]["summary_text"])

    wandb.log({"summarization_after_fine_tuning": table_after_fine_tuning})

if __name__ == "__main__":
    main()
