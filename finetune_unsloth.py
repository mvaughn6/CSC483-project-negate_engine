from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

if not hasattr(torch.amp, "is_autocast_available"):
    torch.amp.is_autocast_available = lambda device_type=None: torch.cuda.is_available()


# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-bnb-4bit",
    max_seq_length=2048,
)

model = FastLanguageModel.get_peft_model(model)

# Use Llama-3.2 chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.2")

# Load and preprocess Orion Weller dataset
raw_ds = load_dataset("orionweller/NevIR", split="train")

def to_conv(ex):
    return {
        "conversations": [
            {"role": "user", "content": ex["q1"]},
            {"role": "assistant", "content": ex["doc1"]},
            {"role": "user", "content": ex["q2"]},
            {"role": "assistant", "content": ex["doc2"]},
        ]
    }

conversations_dataset = raw_ds.map(to_conv, remove_columns=raw_ds.column_names, num_proc=2)

# Apply chat template
dataset = conversations_dataset.map(
    lambda x: {
        "text": tokenizer.apply_chat_template(
            x["conversations"],
            tokenize=False,
            add_generation_prompt=False
        )
    },
    batched=True,
    batch_size=100,
    desc="Formatting conversations"
)

# Training setup
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    dataset_num_proc=2,
    max_seq_length=2048,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=2,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer.train()

# Save final model as GGUF
model.save_pretrained_gguf("ggufmodel", tokenizer, quantization_method="q4_k_m")
