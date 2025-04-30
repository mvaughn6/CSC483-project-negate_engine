from datasets import load_dataset
from tfidf_system import IRSystem
import ollama

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
        if res1 == 1 and res2 == 2:
            points+=1
    
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