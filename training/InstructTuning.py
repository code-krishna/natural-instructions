from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import json
from datasets import Dataset, DatasetDict, load_dataset
import wandb
import torch
from trl import SFTConfig, SFTTrainer
# Before starting training, clear any cached memory
torch.cuda.empty_cache()

# If necessary, adjust memory management settings
torch.backends.cudnn.benchmark = True

def load_custom_dataset(file_path, tokenizer, split_ratio=0.9):
    with open(file_path, 'r') as file:
        data = json.load(file)

    prompts = [x['Prompt'] for x in data]
    answers = [x['response'] for x in data]

    # Tokenize the data and ensure 'input_ids' is properly created
    tokenized_prompts = tokenizer(prompts, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to("cuda")
    tokenized_answers = tokenizer(answers, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to("cuda")

    # Check if 'input_ids' key exists in the tokenized outputs
    if 'input_ids' not in tokenized_prompts or 'input_ids' not in tokenized_answers:
        raise KeyError("Tokenization did not generate 'input_ids'. Check tokenizer settings.")

    # Construct labels, replacing padding token ID with -100
    labels = tokenized_answers['input_ids']
    labels[labels == tokenizer.pad_token_id] = -100

    # Creating a Dataset from these tokenized outputs
    dataset = Dataset.from_dict({
        "input_ids": tokenized_prompts['input_ids'],
        "attention_mask": tokenized_prompts['attention_mask'],
        "labels": labels
    })

    train_test_split = dataset.train_test_split(test_size=1.0 - split_ratio)
    return DatasetDict({'train': train_test_split['train'], 'validation': train_test_split['test']})

model_checkpoint = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Ensure that tokenizer's pad token is correctly set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to("cuda")
    model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to match tokenizer
else:
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to("cuda")

dataset = load_custom_dataset('instruction_tunning_dataset_small.json', tokenizer)

# Set up Trainer arguments
training_args = TrainingArguments(
    output_dir="./llama_sft",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=100,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    report_to="wandb"
)

# Create the SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    max_seq_length=1024,
)

trainer.train()

model.save_pretrained('./finetuned_llama')
tokenizer.save_pretrained('./finetuned_llama')

print("Fine-tuning complete. Model saved to './finetuned_llama'")
wandb.finish()
