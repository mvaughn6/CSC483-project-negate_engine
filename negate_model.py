from datasets import load_dataset
import ollama
import uuid
from nltk.corpus import wordnet as wn
import nltk
# nltk.download("wordnet")
# nltk.download("omw-1.4")

class NegateModel:
    def __init__(self):
        NEG_WORDS = {
            "not", "no", "none", "never",
            "nothing", "nobody", "nowhere", "neither", "nor",
            "can't", "cannot", "dont", "don't", "wont", "won't",
            "shouldnt", "shouldn't", "couldnt", "couldn't",
            "wouldnt", "wouldn't", "mightnt", "mightn't",
            "mustnt", "mustn't",
            "isnt", "isn't", "arent", "aren't",
            "wasnt", "wasn't", "werent", "weren't",
            "doesnt", "doesn't", "didnt", "didn't",
            "hasnt", "hasn't", "havent", "haven't",
            "hadnt", "hadn't",
            "without", "barely", "hardly", "scarcely",
            "rarely", "seldom", "lack", "lacking",
            "absent", "minus"
        }
        # build a dynamic negation list from WordNet
        # seed = ["avoid", "dislike", "refuse", "reject", "deny", "oppose", "hate",
        #         "ignore", "doubt", "decline", "skip", "abstain", "exclude", "hesitate",
        #         "never", "seldom", "rarely"]
        # self.neg_verbs = self._expand_negation_verbs(seed)


    def run_query(self, q, d1, d2):
        
        prompt = self._build_prompt(q, d1, d2)
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
        return resp

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

    def _build_prompt(self, q, d1, d2):
       
        prompt = f"""
            '''INSTRUCTIONS'''\
                You MUST respond with a single number: 1 or 2. Do not provide an explanation\
                    for your reasoning.
            Step 1) Carefully analyze the query delimited by the <q> </q> tags \
                take special care to observe negated terms in the query.
            Step 2) Carefully alalyze the text in document 1 delimited by the <d1> </d1> tag \
                and the text in document 2 delimited by the <d2> </d2> tag.
            Step 3) When you’ve finished analyzing, you must match the query with the most \
                relivent document. Pay special attention to any negation markers when determining \
                the most relivent document.
            Step 4)  When you’ve finished reasoning, return exactly one character:
                `1` if Document 1 matches best\n
                `2` if Document 2 matches best\n

            <q>{q}</q>
            <d1>{d1}</d1>
            <d2>{d2}</d2>
        """

        return prompt