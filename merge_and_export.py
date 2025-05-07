import argparse, os
from unsloth import FastLanguageModel        
from unsloth.save import unsloth_save_pretrained_gguf

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",    required=True,
                   help="HF repo id of *base* model, e.g. unsloth/Llama‑3.2‑3B‑Instruct‑bnb‑4bit")
    p.add_argument("--lora_weights",  required=True,
                   help="Path to LoRA checkpoint dir (checkpoint‑500)")
    p.add_argument("--output_gguf",   required=True,
                   help="Output .gguf path to create")
    return p.parse_args()

def main():
    args = parse()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name       = args.base_model,
        max_seq_length   = 4096,
        load_in_4bit     = True,     
    )
    model.load_adapter(args.lora_weights)

    out_dir  = os.path.dirname(args.output_gguf)
    prefix   = os.path.splitext(os.path.basename(args.output_gguf))[0]
    os.makedirs(out_dir, exist_ok=True)

    print(f" Writing merged model to {args.output_gguf} …")
    unsloth_save_pretrained_gguf(
        model               = model,
        tokenizer           = tokenizer,
        save_directory      = out_dir,
        filename_prefix     = prefix,     
        quantization_method = ["q4_0"],    
    )
    print(" GGUF saved!")

if __name__ == "__main__":
    main()
