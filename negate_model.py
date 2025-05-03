from datasets import load_dataset
import ollama


class NegateModel:

    def __init__(self):
        self.client = ollama.Client()

        
    def run_query(self, q, d1, d2):
        prompt = self._get_query(q,d1,d2)
        self.client = ollama.Client()
        #model = "llama3.2"
        model = "phi4-mini"
        #model = "llama3.2:1b"

        # response = self.client.generate(model=model, prompt=prompt, options={"temperature": 0, "max_new_tokens": 5})
        response = self.client.generate(model=model, prompt=prompt, options={"temperature": 0})

        # return response["response"].strip()
        return self._clean_response(response["response"])
      
            

    def _get_query(self, q, d1, d2):
        # ——— Few-Shot Examples ———
        examples = (
            "```EXAMPLES\n"
            "Example 1:\n"
            "Query: Which animals are friendly toward humans?\n"
            "Document 1: Dogs have been bred for companionship and are very friendly toward people.\n"
            "Document 2: Wolves are wild animals and often avoid human contact.\n"
            "Answer: 1\n\n"

            "Example 2:\n"
            "Query: Which fruits are not safe to eat?\n"
            "Document 1: Ripe mangoes and bananas are edible and nutritious.\n"
            "Document 2: Certain wild berries contain toxins and cause illness if consumed.\n"
            "Answer: 2\n"
            "```\n\n"
        )

        # ——— Instructions / Data / Output ———
        instr = (
            "```INSTRUCTIONS\n"
            "1. You will see a Query and two Documents (1 and 2).\n"
            "2. Negation may appear explicitly (e.g. “not,” “never”) or implicitly (e.g. “avoid,” antonyms).\n"
            "3. Internally think step by step about how each document’s meaning aligns with—or contradicts—the Query’s intent.\n"
            "4. When you’ve finished reasoning, output **exactly one digit** with NO explanation:\n"
            "     `1` if Document 1 matches the Query best\n"
            "     `2` if Document 2 matches the Query best\n"
            "```\n\n"
        )

        data = (
            "```DATA\n"
            f"Query: {q}\n"
            f"Document 1: {d1}\n"
            f"Document 2: {d2}\n"
            "```\n\n"
        )


        return examples + instr + data
    

    # def _get_query(self, q, d1, d2):
        # examples = (
        #     "Example 1:\n"
        #     "Query: Which animals are friendly toward humans?\n"
        #     "Document 1: Dogs are known for their friendly behavior toward humans.\n"
        #     "Document 2: Some wild animals like wolves are not considered friendly toward humans.\n"
        #     "Answer: 1\n\n"
            
        #     "Example 2:\n"
        #     "Query: What fruits are safe for humans to eat?\n"
        #     "Document 1: Apples and bananas are commonly eaten safely by humans.\n"
        #     "Document 2: Some wild berries are poisonous and unsafe for consumption.\n"
        #     "Answer: 1\n\n"
        # )
        
        # prompt = (
        #     "```INSTRUCTIONS\n"
        #     "1. Read the Query, Document 1, and Document 2.\n"
        #     "2. Look for any words or phrases that indicate negation or opposite meaning—both explicit (e.g., “not,” “never”) and implicit (e.g., “avoid,” “refuse,” antonyms or contrary statements).\n"
        #     "3. Think step by step about how each document’s meaning aligns with—or contradicts—the Query’s intent.\n"
        #     "4. When you’ve finished reasoning, output exactly one character (no other text):\n"
        #      "   - `1` if Document 1 matches best\n"
        #      "   - `2` if Document 2 matches best\n"
        #     "5. Do **not** include any explanation or other text.\n"
        #     "```\n\n"
        #     "```DATA\n"
        #     f"Query: {q}\n"
        #     f"Document 1: {d1}\n"
        #     f"Document 2: {d2}\n"
        #     "```\n\n"
        #     "```OUTPUT\n"
        #     "Answer:\n"
        #     "```\n"
        #     )
        
        # instr_data = (
        #         "```INSTRUCTIONS\n"
        #         "1. You will receive a Query and two candidate Documents (1 and 2).\n"
        #         "2. Negation may appear explicitly (e.g. “not,” “never”) or implicitly (e.g. “avoid,” antonyms).\n"
        #         "3. Internally think step by step about which document truly matches the Query’s intent.\n"
        #         "4. Then output exactly one digit—with no explanation:\n"
        #         "   - `1` if Document 1 matches best\n"
        #         "   - `2` if Document 2 matches best\n"
        #         "```\n\n"

        #         "```DATA\n"
        #         f"Query: {q}\n"
        #         f"Document 1: {d1}\n"
        #         f"Document 2: {d2}\n"
        #         "```\n\n"

        #         "```OUTPUT\n"
        #         "Answer:\n"
        #         "```"
        #     )

        # return instr_data

    # def _get_query(self, q, d1, d2):

    #     # ——— Instructions & Data & Output ———
    #     instr_data = (
    #         "```INSTRUCTIONS\n"
    #         "1. You will receive a Query and two candidate Documents (1 and 2).\n"
    #         "2. Negation may appear explicitly (e.g. “not,” “never”) or implicitly (e.g. “avoid,” antonyms).\n"
    #         "3. Internally think step by step about which document truly matches the Query’s intent.\n"
    #         "4. Then output exactly one digit—with no explanation:\n"
    #         "     1  if Document 1 matches best\n"
    #         "     2  if Document 2 matches best\n"
    #         "```\n\n"

    #         "```DATA\n"
    #         f"Query: {q}\n"
    #         f"Document 1: {d1}\n"
    #         f"Document 2: {d2}\n"
    #         "```\n\n"

    #         "```OUTPUT\n"
    #         "Answer:\n"
    #         "```"
    #     )

    #     return instr_data
    
    def _clean_response(self, resp):
        resp = resp.strip()
        if resp.startswith("1"):
            return "1"
        elif resp.startswith("2"):
            return "2"
        else:
            return "unknown"




