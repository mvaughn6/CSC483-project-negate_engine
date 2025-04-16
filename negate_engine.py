from datasets import load_dataset
import ollama



def main():
    # Load the dataset using its identifier on Hugging Face
    dataset = load_dataset("orionweller/NevIR")

    train_data = dataset["train"]
    # print(train_data[0]["q1"])  # Print the first example from the training split
    # print(train_data[0]["q2"])  # Print the first example from the training split
    # print(train_data[0]["doc1"])  # Print the first example from the training split
    # print(train_data[0]["doc2"])  # Print the first example from the training split


    client = ollama.Client()

    model = "gemma3"
    prompt = "hows it going?"

    response = client.generate(model=model, prompt=prompt)

    print(response["response"])



if __name__ == "__main__":
    main()