from datasets import load_dataset
from tfidf_system import IRSystem
import ollama
from negate_model import NegateModel

from negate_engine import *

# Load the dataset using its identifier on Hugging Face
dataset = load_dataset("orionweller/NevIR")
test_data = dataset["test"]


def test_llm(num_tests):
    points = 0

    for i in range(num_tests):
        q1 = test_data[i]["q1"] 
        q2 = test_data[i]["q2"]  
        d1 = test_data[i]["doc1"] 
        d2 = test_data[i]["doc2"]

        llm = NegateModel()
        res1 = llm.run_query(q1, d1, d2)
        res2 = llm.run_query(q2, d1, d2)
        if int(res1) == 1 and int(res2) == 2:
            points+=1
        print("Test: " + str(i+1))
        print("Q1 Res: " + res1)
        print("Q2 Res: " + res2 + "\n")
    
    score = points / num_tests
    percentage = score * 100
    return percentage

def test_tfidf(num_tests):
    points = 0
    for i in range(num_tests):
        q1 = test_data[i]["q1"] 
        q2 = test_data[i]["q2"]  
        d1 = test_data[i]["doc1"] 
        d2 = test_data[i]["doc2"]

        docs = "1 " + d1 + "\n" + "2 " + d2 + "\n"

        with open("doc-coll.txt", "w", encoding="utf-8") as file:
            file.write(docs)

        ir = IRSystem(open("doc-coll.txt", "r", encoding="UTF-8"))
        res1 = ir.run_query(query=q1)
        res2 = ir.run_query(query=q2)
        if res1 == 1 and res2 == 2:
            points+=1
        # print("Test: " + str(i+1))
        # print("Q1 Res: " + str(ir.run_query(query=q1)))
        # print("Q2 Res: " + str(ir.run_query(query=q2)) + "\n")

    score = points / num_tests
    percentage = score * 100
    return percentage


def test_rand_full():
    num_tests = len(test_data)
    points = 0
    for i in range(num_tests):
        res1 = random_query_res()
        res2 = random_query_res()
        if res1 == 1 and res2 == 2:
            points+=1

    score = points / num_tests
    percentage = score * 100
    return percentage


def test_rand_onehundred():
    num_tests = 100
    score = 0
    for _ in range(num_tests):
        score += test_rand_full()
    return score / num_tests


def test_rand_onethousand():
    num_tests = 1000
    score = 0
    for _ in range(num_tests):
        score += test_rand_full()
    return score / num_tests


def random_query_res():
    return random.randint(1, 2)






# from pytest import approx
# from datasets import load_dataset
# import ollama

# import negate_engine

# # ir = tfidf_engine.IRSystem(open("wiki-small.txt", encoding = "UTF-8"))


# def test1():
#     dataset = load_dataset("orionweller/NevIR")

#     train_data = dataset["train"]
#     q1 = train_data[0]["q1"]  # Print the first example from the training split
#     q2 = train_data[0]["q2"]  # Print the first example from the training split
#     d1 = train_data[0]["doc1"]  # Print the first example from the training split
#     d2 = train_data[0]["doc2"]  # Print the first example from the training split

#     prompt1 = "Based on this query: " + q1 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"
#     prompt2 = "Based on this query: " + q2 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"

#     f = negate_engine.run_query(prompt1)
#     s = negate_engine.run_query(prompt2)

#     print(f)
#     print(s)

#     assert f == "1"
#     assert s == "2"

# def test2():
#     dataset = load_dataset("orionweller/NevIR")

#     train_data = dataset["train"]
#     q1 = train_data[1]["q1"]  # Print the first example from the training split
#     q2 = train_data[1]["q2"]  # Print the first example from the training split
#     d1 = train_data[1]["doc1"]  # Print the first example from the training split
#     d2 = train_data[1]["doc2"]  # Print the first example from the training split

#     prompt1 = "Based on this query: " + q1 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"
#     prompt2 = "Based on this query: " + q2 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"

#     f = negate_engine.run_query(prompt1)
#     s = negate_engine.run_query(prompt2)

#     print(f)
#     print(s)

#     assert f == "1"
#     assert s == "2"


# def test3():
#     dataset = load_dataset("orionweller/NevIR")

#     train_data = dataset["train"]
#     q1 = train_data[2]["q1"]  # Print the first example from the training split
#     q2 = train_data[2]["q2"]  # Print the first example from the training split
#     d1 = train_data[2]["doc1"]  # Print the first example from the training split
#     d2 = train_data[2]["doc2"]  # Print the first example from the training split

#     prompt1 = "Based on this query: " + q1 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"
#     prompt2 = "Based on this query: " + q2 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"

#     f = negate_engine.run_query(prompt1)
#     s = negate_engine.run_query(prompt2)

#     print(f)
#     print(s)

#     assert f == "1"
#     assert s == "2"


# def test4():
#     dataset = load_dataset("orionweller/NevIR")

#     train_data = dataset["train"]
#     q1 = train_data[3]["q1"]  # Print the first example from the training split
#     q2 = train_data[3]["q2"]  # Print the first example from the training split
#     d1 = train_data[3]["doc1"]  # Print the first example from the training split
#     d2 = train_data[3]["doc2"]  # Print the first example from the training split

#     prompt1 = "Based on this query: " + q1 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"
#     prompt2 = "Based on this query: " + q2 + " which document relates best to the query from the perspective of negation? " + "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"

#     f = negate_engine.run_query(prompt1)
#     s = negate_engine.run_query(prompt2)

#     print(f)
#     print(s)

#     assert f == "1"
#     assert s == "2"

