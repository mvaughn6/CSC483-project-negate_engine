# prepare_negation_data.py
import json
from datasets import load_dataset

def main():
    ds = load_dataset("orionweller/NevIR", split="train")
    with open("negation_train.jsonl", "w") as out:
        for ex in ds:
            # Query q1 → label “1”
            inp1 = (
                f"Query: {ex['q1']} "
                f"Document 1: {ex['doc1']} "
                f"Document 2: {ex['doc2']}"
            )
            out.write(json.dumps({"input": inp1, "output": "1"}) + "\n")
            # Query q2 → label “2”
            inp2 = (
                f"Query: {ex['q2']} "
                f"Document 1: {ex['doc1']} "
                f"Document 2: {ex['doc2']}"
            )
            out.write(json.dumps({"input": inp2, "output": "2"}) + "\n")

if __name__ == "__main__":
    main()
