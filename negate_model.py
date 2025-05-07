from datasets import load_dataset
import ollama
import re
from functools import lru_cache

class NegateModel:
    def __init__(self, model_name: str = "mistral", temperature: float = 0.0, max_new_tokens: int = 1):
        self.client = ollama.Client()
        self.model = model_name
        self.options = {
            "max_new_tokens": max_new_tokens,
            "stop": ["\n"]
        }

    @lru_cache(maxsize=None)
    def run_query(self, q: str, d1: str, d2: str) -> str:
        prompt = self._build_prompt(q, d1, d2)
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options=self.options
        )
        raw = response.get("response", "")
        cleaned = self._clean_response(raw)

        if cleaned not in ("1", "2"):
            retry_prompt = prompt + "\nAnswer:"  
            response2 = self.client.generate(
                model=self.model,
                prompt=retry_prompt,
                options=self.options
            )
            raw2 = response2.get("response", "")
            cleaned = self._clean_response(raw2)

        return cleaned

    def _build_prompt(self, q: str, d1: str, d2: str) -> str:
        """
        Constructs a few-shot prompt with instructions and examples.
        """
        examples = (
            "Example 1:\n"
            "Query: Which animals are friendly toward humans?\n"
            "Document 1: Dogs have been bred for companionship and are very friendly toward people.\n"
            "Document 2: Wolves are wild animals and often avoid human contact.\n"
            "Answer: 1\n\n"

            "Example 2:\n"
            "Query: Which fruits are not safe to eat?\n"
            "Document 1: Ripe mangoes and bananas are edible and nutritious.\n"
            "Document 2: Certain wild berries contain toxins and cause illness if consumed.\n"
            "Answer: 2\n\n"
        )
        instructions = (
            "You will see a Query and two Documents (1 and 2). "
            "Negation may appear explicitly (e.g., 'not', 'never') or implicitly (e.g., 'avoid', antonyms). "
            "Think step-by-step about how each document’s meaning aligns with—or contradicts—the Query’s intent. "
            "Then output exactly one digit ('1' or '2') with no additional text.\n\n"
        )
        data = (
            f"Query: {q}\n"
            f"Document 1: {d1}\n"
            f"Document 2: {d2}\n"
        )
        return instructions + data

    # def _build_prompt(self, q, d1, d2):
    #     examples = f"""
    #         '''Example 1'''
    #         <q>Which mayor did less vetoing than anticipated?</q>
    #         Document 1: In his first year as mayor, Medill received very little legislative resistance from the Chicago City Council. While he vetoed what was an unprecedented eleven City Council ordinances that year, most narrowly were involved with specific financial practices considered wasteful and none of the vetoes were overridden. He used his new powers to appoint the members of the newly constituted Chicago Board of Education and the commissioners of its constituted public library. His appointments were approved unanimously by the City Council.</d1>
    #         Document 2: In his first year as mayor, Medill received very little legislative resistance from the Chicago City Council. While some expected an unprecedented number of vetoes, in actuality he only vetoed eleven City Council ordinances that year, and most of those were narrowly involved with specific financial practices he considered wasteful and none of the vetoes were overridden. He used his new powers to appoint the members of the newly constituted Chicago Board of Education and the commissioners of its constituted public library. His appointments were approved unanimously by the City Council.</d2>
    #         Response: 2

    #          '''Example 2'''
    #         <q>Which mayor did more vetoing than anticipated?</q>
    #         <d1>In his first year as mayor, Medill received very little legislative resistance from the Chicago City Council. While he vetoed what was an unprecedented eleven City Council ordinances that year, most narrowly were involved with specific financial practices considered wasteful and none of the vetoes were overridden. He used his new powers to appoint the members of the newly constituted Chicago Board of Education and the commissioners of its constituted public library. His appointments were approved unanimously by the City Council.</d1>
    #         <d2>In his first year as mayor, Medill received very little legislative resistance from the Chicago City Council. While some expected an unprecedented number of vetoes, in actuality he only vetoed eleven City Council ordinances that year, and most of those were narrowly involved with specific financial practices he considered wasteful and none of the vetoes were overridden. He used his new powers to appoint the members of the newly constituted Chicago Board of Education and the commissioners of its constituted public library. His appointments were approved unanimously by the City Council.</d2>
    #         Response: 1
    #     """
    #     # '''Example 3'''
    #     #     <q>For what is it not possible to count the number of steps?
    #     #     <d1>It is impossible to count the number of steps of an algorithm on all possible inputs. As the complexity generally increases with the size of the input, the complexity is typically expressed as a function of the size (in bits) of the input, and therefore, the complexity is a function of. However, the complexity of an algorithm may vary dramatically for different inputs of the same size. Therefore, several complexity functions are commonly used.</d1>
    #     #     <d2>It is possible to count the approximate number of steps of an algorithm on all possible inputs. As the complexity generally increases with the size of the input, the complexity is typically expressed as a function of the size (in bits) of the input, and therefore, the complexity is a function of. However, the complexity of an algorithm may vary dramatically for different inputs of the same size. Therefore, several complexity functions are commonly used.</d2>
    #     #     Response: 1

    #     prompt = f"""
    #         '''INSTRUCTIONS'''\
    #             You MUST respond with a single number: 1 or 2. Do not provide an explaination\
    #                 for your reasoning.
    #         Step 1) Review the provided examples delimited by the <ex> </ex> tags
    #         Step 2) Carefully analyze the query delimited by the <q> </q> tags \
    #             take special care to observe negated terms in the query.
    #         Step 3) Carefully alalyze the text in document 1 delimited by the <d1> </d1> tag \
    #             and the text in document 2 delimited by the <d2> </d2> tag.
    #         Step 4) When you’ve finished analyzing, you must match the query with the most \
    #             relivent document. Pay special attention to any negation markers when determining \
    #             the most relivent document.
    #         Step 5)  When you’ve finished reasoning, return exactly one character:
    #             `1` if Document 1 matches best\n
    #             `2` if Document 2 matches best\n

    #         <ex>{examples}</ex>
    #         <q>{q}</q>
    #         <d1>{d1}</d1>
    #         <d2>{d2}</d2>
    #     """

    #     return prompt

    def _clean_response(self, resp: str) -> str:
        match = re.search(r"[12]", resp)
        return match.group(0) if match else "2"

   

