from datasets import load_dataset
import ollama
from test_negation import *
import random


# def run_query(prompt):
#     client = ollama.Client()
#     model = "gemma3:1b"
#     response = client.generate(model=model, prompt=prompt)
#     return response["response"].strip()

def run_random_tests():
    print("Random Scores:")
    score = test_rand_full()
    print(f"\tFull Test Suit : {score:.2f}%")
    score = test_rand_onehundred()
    print(f"\tFull Test 100x : {score:.2f}%")
    score = test_rand_onethousand()
    print(f"\tFull Test 1000x: {score:.2f}%")

def run_tfidf_tests():
    print("\n\ntf-idf Scores:")
    score = test_tfidf(100)
    print(f"\tTest 100 queries : {score:.2f}%")
    score = test_tfidf(1000)
    print(f"\tTest 1000 queries: {score:.2f}%")
    score = test_tfidf(1380)
    print(f"\tTest All queries : {score:.2f}%")

def run_llm_tests():
    print("\n\nLLM Scores:")
    score = test_llm(10)
    print(f"\tTest 10 queries : {score:.2f}%")

def main():
    # run_random_tests()
    # run_tfidf_tests()
    run_llm_tests()

if __name__ == "__main__":
    main()