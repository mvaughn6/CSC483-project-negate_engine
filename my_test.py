from datasets import load_dataset
import ollama
from negate_model import NegateModel

def main():
    # Load the dataset using its identifier on Hugging Face
    dataset = load_dataset("orionweller/NevIR")

    train_data = dataset["test"]
    q1 = train_data[0]["q1"]  
    q2 = train_data[0]["q2"]  
    d1 = train_data[0]["doc1"]  
    d2 = train_data[0]["doc2"]  

    m = NegateModel()
    print("set: 0")
    print(m.run_query(q1,d1,d2))
    print(m.run_query(q2,d1,d2))
    print("--------------------------")

if __name__ == "__main__":
    main()