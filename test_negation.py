from pytest import approx
from datasets import load_dataset
import ollama

import negate_engine

# ir = tfidf_engine.IRSystem(open("wiki-small.txt", encoding = "UTF-8"))


def test1():
    dataset = load_dataset("orionweller/NevIR")

    train_data = dataset["train"]
    q1 = train_data[0]["q1"]  # Print the first example from the training split
    q2 = train_data[0]["q2"]  # Print the first example from the training split
    d1 = train_data[0]["doc1"]  # Print the first example from the training split
    d2 = train_data[0]["doc2"]  # Print the first example from the training split

    prompt1 = "Based on this query: " + q1 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"
    prompt2 = "Based on this query: " + q2 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"

    f = negate_engine.run_query(prompt1)
    s = negate_engine.run_query(prompt2)

    print(f)
    print(s)

    assert f == "1"
    assert s == "2"

def test2():
    dataset = load_dataset("orionweller/NevIR")

    train_data = dataset["train"]
    q1 = train_data[1]["q1"]  # Print the first example from the training split
    q2 = train_data[1]["q2"]  # Print the first example from the training split
    d1 = train_data[1]["doc1"]  # Print the first example from the training split
    d2 = train_data[1]["doc2"]  # Print the first example from the training split

    prompt1 = "Based on this query: " + q1 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"
    prompt2 = "Based on this query: " + q2 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"

    f = negate_engine.run_query(prompt1)
    s = negate_engine.run_query(prompt2)

    print(f)
    print(s)

    assert f == "1"
    assert s == "2"


def test3():
    dataset = load_dataset("orionweller/NevIR")

    train_data = dataset["train"]
    q1 = train_data[2]["q1"]  # Print the first example from the training split
    q2 = train_data[2]["q2"]  # Print the first example from the training split
    d1 = train_data[2]["doc1"]  # Print the first example from the training split
    d2 = train_data[2]["doc2"]  # Print the first example from the training split

    prompt1 = "Based on this query: " + q1 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"
    prompt2 = "Based on this query: " + q2 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"

    f = negate_engine.run_query(prompt1)
    s = negate_engine.run_query(prompt2)

    print(f)
    print(s)

    assert f == "1"
    assert s == "2"


def test4():
    dataset = load_dataset("orionweller/NevIR")

    train_data = dataset["train"]
    q1 = train_data[3]["q1"]  # Print the first example from the training split
    q2 = train_data[3]["q2"]  # Print the first example from the training split
    d1 = train_data[3]["doc1"]  # Print the first example from the training split
    d2 = train_data[3]["doc2"]  # Print the first example from the training split

    prompt1 = "Based on this query: " + q1 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"
    prompt2 = "Based on this query: " + q2 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"

    f = negate_engine.run_query(prompt1)
    s = negate_engine.run_query(prompt2)

    print(f)
    print(s)

    assert f == "1"
    assert s == "2"

