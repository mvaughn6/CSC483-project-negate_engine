from datasets import load_dataset
import ollama


def run_query(prompt):
    client = ollama.Client()
    model = "gemma3"
    response = client.generate(model=model, prompt=prompt)
    return response["response"].strip()


def main():
    # Load the dataset using its identifier on Hugging Face
    dataset = load_dataset("orionweller/NevIR")

    train_data = dataset["test"]
    q1 = train_data[0]["q1"]  # Print the first example from the training split
    q2 = train_data[0]["q2"]  # Print the first example from the training split
    d1 = train_data[0]["doc1"]  # Print the first example from the training split
    d2 = train_data[0]["doc2"]  # Print the first example from the training split

    prompt = "Based on this query: " + q1 + " which document relates best to the query: " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"


if __name__ == "__main__":
    main()