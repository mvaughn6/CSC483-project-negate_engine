from datasets import load_dataset
import ollama

from negate_model import NegateModel



def main():
    # Load the dataset using its identifier on Hugging Face
    dataset = load_dataset("orionweller/NevIR")

    train_data = dataset["test"]
    q1 = train_data[0]["q1"]  # Print the first example from the training split
    q2 = train_data[0]["q2"]  # Print the first example from the training split
    d1 = train_data[0]["doc1"]  # Print the first example from the training split
    d2 = train_data[0]["doc2"]  # Print the first example from the training split

    m = NegateModel()
    print("set: 0")
    print(m.run_query(q1,d1,d2))
    print(m.run_query(q2,d1,d2))
    print("--------------------------")
   

       



    # q1 = "Which animals are considered friendly toward humans?"
    # q2 = "Which animals are not considered friendly toward humans?"
    # d1 = "Dogs have been domesticated for thousands of years and are known for their friendly behavior toward humans."
    # d2 = "Some wild animals, like wolves, are often aggressive and are not considered friendly toward humans."

    # q1 = "Which fruits are safe for human consumption?"
    # q2 = "Which fruits are not safe for human consumption?"
    # d1 = "Apples, bananas, and strawberries are examples of fruits that are safe and healthy for humans to eat."
    # d2 = "Certain wild berries contain toxins and are unsafe for human consumption."



    # print("query 1: " + q1)
    # print("query 2: " + q2)
    # print("doc 1: " + d1)
    # print("doc 2: " + d2)






if __name__ == "__main__":
    main()