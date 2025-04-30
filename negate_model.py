# from datasets import load_dataset
# import ollama


# class NegateModel:

#     def __init__(self):
#         pass
        
#     def run_query(self, q, d1, d2):
#         prompt = self._get_query(q,d1,d2)
#         client = ollama.Client()
#         # model = "llama3.2"
#         model = "phi4-mini"

#         response = client.generate(model=model, prompt=prompt, options={"temperature": 0})
#         # return response["response"].strip()
#         return self._clean_response(response["response"])
    
#     # def _get_query(self, q, d1, d2):
#     #     # prompt = "Based on this query: " + q + " which document relates best to the query: " + \
#     #     #     "doc1: " +  d1 + "or doc2: " + d2 + "please only return 1 for doc1 or only return 2 for doc2"
#     #     prompt = (
#     #         "Task: Compare Document 1 and Document 2 to the Query.\n"
#     #         "One document may contradict the query using negation or opposite meaning.\n"
#     #         "Which document matches the query better?\n\n"
#     #         f"Query: {q}\n"
#     #         f"Document 1: {d1}\n"
#     #         f"Document 2: {d2}\n\n"
#     #         "Return ONLY the number 1 if Document 1 matches better or 2 if Document 2 matches better. Do not explain. Do not return anything else."
#     #     )
#     #     return prompt
    
#     # def _get_query(self, q, d1, d2):
#     #     prompt = (
#     #     "Task: Carefully compare Document 1 and Document 2 to the Query.\n"
#     #     "Focus especially on words that indicate time, history, negation, or previous conditions.\n"
#     #     "Which document correctly answers the query based on meaning? Be precise.\n\n"
#     #     f"Query: {q}\n"
#     #     f"Document 1: {d1}\n"
#     #     f"Document 2: {d2}\n\n"
#     #     "Return ONLY the number 1 if Document 1 matches better or 2 if Document 2 matches better. Do not explain. Do not return anything else."
#     #     )
#     #     return prompt

#     def _get_query(self, q, d1, d2):
#         examples = (
#             "Example 1:\n"
#             "Query: Which animals are friendly toward humans?\n"
#             "Document 1: Dogs are known for their friendly behavior toward humans.\n"
#             "Document 2: Some wild animals like wolves are not considered friendly toward humans.\n"
#             "Answer: 1\n\n"
            
#             "Example 2:\n"
#             "Query: What fruits are safe for humans to eat?\n"
#             "Document 1: Apples and bananas are commonly eaten safely by humans.\n"
#             "Document 2: Some wild berries are poisonous and unsafe for consumption.\n"
#             "Answer: 1\n\n"
#         )

#         prompt = (
#             examples +
#             "Now for the real task:\n"
#             "Carefully compare Document 1 and Document 2 to the Query.\n"
#             "Focus especially on words that indicate time, history, negation, safety, friendliness, or previous conditions.\n"
#             "Which document correctly answers the query based on meaning? Be precise.\n\n"
#             f"Query: {q}\n"
#             f"Document 1: {d1}\n"
#             f"Document 2: {d2}\n\n"
#             "Return ONLY the number 1 if Document 1 matches better or 2 if Document 2 matches better. Do not explain. Do not return anything else."
#         )
#         return prompt
    
#     def _clean_response(self, resp):
#         resp = resp.strip()
#         if resp.startswith("1"):
#             return "1"
#         elif resp.startswith("2"):
#             return "2"
#         else:
#             return "unknown"




from datasets import load_dataset
import ollama
import uuid
from nltk.corpus import wordnet as wn
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")


# Make sure you’ve run these once:
# nltk.download('wordnet')
# nltk.download('omw-1.4')

class NegateModel:
    def __init__(self):
        # build a dynamic negation list from WordNet
        seed = ["avoid", "dislike", "refuse", "reject", "deny", "oppose", "hate",
                "ignore", "doubt", "decline", "skip", "abstain", "exclude", "hesitate",
                "never", "seldom", "rarely"]
        self.neg_verbs = self._expand_negation_verbs(seed)

    def _expand_negation_verbs(self, seeds):
        expanded = set(seeds)
        for verb in seeds:
            for syn in wn.synsets(verb, pos=wn.VERB):
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ").lower()
                    if len(name.split()) <= 2:
                        expanded.add(name)
        return expanded

    def emphasize_negation(self, text):
        t = text.lower()
        for verb in sorted(self.neg_verbs, key=len, reverse=True):
            if verb in t:
                # wrap the longest matches first
                text = text.replace(verb, f"[NEG:{verb.upper().replace(' ', '_')}]")
        return text

    def run_query(self, q, d1, d2):
        # 1) tag docs
        d1_proc = self.emphasize_negation(d1)
        d2_proc = self.emphasize_negation(d2)

        # 2) build & log prompt
        prompt = self._get_query(q, d1_proc, d2_proc)
        # optional: append a unique ID to avoid any cache
        prompt += f"\n\nSession-ID: {uuid.uuid4()}\n"

        client = ollama.Client()
        resp = client.generate(
            model="phi4-mini",
            prompt=prompt, options={
                "temperature": 0,      
                "max_new_tokens": 20 
            }
        )["response"]

        # 3) clean and fallback
        ans = self._clean_response(resp)
        if ans == "unknown":
            return self._heuristic_fallback(q, d1_proc, d2_proc)
        return ans

    def _clean_response(self, resp: str) -> str:
        r = resp.strip().splitlines()[0]
        if r.startswith("1"):
            return "1"
        if r.startswith("2"):
            return "2"
        return "unknown"

    def _heuristic_fallback(self, q, d1, d2):
        # simple: if query negated, pick the doc with most neg markers; else pick least
        neg_in_q = "not" in q.lower() or any(v in q.lower() for v in ["never", "avoid", "dislike"])
        count1 = sum(d1.count(f"[NEG:{v.upper().replace(' ', '_')}]") for v in self.neg_verbs)
        count2 = sum(d2.count(f"[NEG:{v.upper().replace(' ', '_')}]") for v in self.neg_verbs)
        if neg_in_q:
            return "1" if count1 > count2 else "2"
        else:
            return "1" if count1 < count2 else "2"

    # def _get_query(self, q, d1, d2):
    #     # 2-shot few-shot examples
    #     examples = (
    #         "Example 1:\n"
    #         "Query: Which animals are friendly toward humans?\n"
    #         "Document 1: Dogs are known for their friendly behavior toward humans.\n"
    #         "Document 2: Some wild animals like wolves are [NEG:NOT] considered friendly toward humans.\n"
    #         "Answer: 1\n\n"
            
    #         "Example 2:\n"
    #         "Query: What fruits are safe for humans to eat?\n"
    #         "Document 1: Apples and bananas are commonly eaten safely by humans.\n"
    #         "Document 2: Some wild berries are [NEG:POISONOUS] and unsafe for consumption.\n"
    #         "Answer: 1\n\n"
    #     )

    #     instruction = (
    #         "Now for the real task:\n"
    #         "First, think step by step about how each document aligns with the query.\n"
    #         "Then return ONLY the number 1 or 2 (no explanation, no other text).\n\n"
    #     )

    #     return (
    #         examples
    #         + instruction
    #         + f"Query: {q}\n"
    #         + f"Document 1: {d1}\n"
    #         + f"Document 2: {d2}\n\n"
    #         + "Answer:"
    #     )

    def _get_query(self, q, d1, d2):
        # examples = (
        #     "Example 1:\n"
        #     "Query: Which animals are friendly toward humans?\n"
        #     "Document 1: Dogs are known for their friendly behavior toward humans.\n"
        #     "Document 2: Some wild animals like wolves are [NEG:NOT] considered friendly toward humans.\n"
        #     "Answer: 1\n\n"
        #     "the correct answer is 1 here because the query does not express negation which matches document 1"

        #     "Example 2:\n"
        #     "Query: What fruits are not safe for humans to eat?\n"
        #     "Document 1: Apples and bananas are commonly eaten safely by humans.\n"
        #     "Document 2: Some wild berries are [NEG:POISONOUS] and unsafe for consumption.\n"
        #     "Answer: 2\n\n"
        #     "the correct answer is 2 here because the query is negated and document 2 expresses negation"
        # )

        prompt = (
            "```INSTRUCTIONS\n"
            "1. Carefully compare the Query with Document 1 and Document 2.\n"
            "2. Think step by step about how each document aligns with the Query’s intent.\n"
            "3. Pay special attention to any negation markers (e.g., [NEG:…]) or temporal cues (“previously,” “newly”).\n"
            "4. When you’ve finished reasoning, return exactly one character:\n"
            "   - `1` if Document 1 matches best\n"
            "   - `2` if Document 2 matches best\n"
            "   - `3` if you are unsure and don't feel confident in your answer"
            "5. Do **not** include any explanation or other text.\n"
            "```\n\n"
            "```DATA\n"
            f"Query: {q}\n"
            f"Document 1: {d1}\n"
            f"Document 2: {d2}\n"
            "```\n\n"
            "```OUTPUT\n"
            "Answer:\n"
            "```\n"
        )

        return prompt
