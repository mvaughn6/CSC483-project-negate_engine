import sys
import logging
from multiprocessing import freeze_support

import datasets
from datasets import load_dataset
from peft import LoraConfig
import torch
import transformers
from trl import SFTConfig, SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)

def main():
    ###################
    # SFT + PEFT Config
    ###################
    sft_config = SFTConfig(
        output_dir="./checkpoints",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=10,
        fp16=True,
        max_length=2048,
        packing=True,
    )

    # Add target_modules so PEFT knows which Linear layers to inject LoRA into
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    ###################
    # Logging setup
    ###################
    logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s —   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    log_level = sft_config.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning("***** Running SFT training *****")
    logger.info(f"SFTConfig: {sft_config}")
    logger.info(f"PEFTConfig: {peft_config}")

    ###################
    # Model & Tokenizer
    ###################
    checkpoint = "microsoft/Phi-4-mini-instruct"
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
        # attn_implementation="flash_attention_2",  # enable if you’ve installed flash-attn
    )
    model     = AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.model_max_length = 2048
    tokenizer.pad_token        = tokenizer.unk_token
    tokenizer.pad_token_id     = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    tokenizer.padding_side     = "right"

    ###################
    # Data Prep
    ###################
    def apply_chat_template(example):
        messages = [
            {"role": "user",      "content": example["q1"]},
            {"role": "assistant", "content": example["q2"]},
        ]
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return example

    train_ds, test_ds = load_dataset("orionweller/NevIR", split=["train", "test"])
    cols            = train_ds.column_names

    processed_train = train_ds.map(
        apply_chat_template,
        remove_columns=cols,
        num_proc=10,
        desc="Preprocessing train",
    )
    processed_test = test_ds.map(
        apply_chat_template,
        remove_columns=cols,
        num_proc=10,
        desc="Preprocessing test",
    )

    ###################
    # SFT Trainer
    ###################
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=processed_train,
        eval_dataset=processed_test,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    ###################
    # Evaluation
    ###################
    tokenizer.padding_side = "left"
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(processed_test)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    trainer.save_model(sft_config.output_dir)


if __name__ == "__main__":
    freeze_support()  # safe multiprocessing.spawn on Windows
    main()
