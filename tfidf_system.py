from collections import defaultdict
import math

# --- Negation annotation ---
NEG_CUES = {
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
NEG_WINDOW = 10

def annotate_negation(tokens):
    out = []
    negate = False
    cnt = 0

    for t in tokens:
        if t in NEG_CUES:
            negate = True
            cnt = NEG_WINDOW
            out.append(t)
        else:
            if negate and cnt > 0:
                out.append("NOT_" + t)
                cnt -= 1
                if cnt == 0:
                    negate = False
            else:
                out.append(t)
    return out


class IRSystem:
    def __init__(self, f):
        self._tf = defaultdict(lambda: defaultdict(float))
        self._df = defaultdict(int)
        self._number_of_docs = 0

        for line in f:
            # tokenize and annotate negation
            tokens = line.lower().split()
            tokens = annotate_negation(tokens)

            # first token is document ID
            doc_id = int(tokens[0])

            # raw term frequencies
            for token in tokens[1:]:
                if token != '-':
                    self._tf[doc_id][token] += 1

            # document frequencies and weighted tf
            for token in set(tokens[1:]):
                if token != '-':
                    self._df[token] += 1
                    raw = self._tf[doc_id][token]
                    self._tf[doc_id][token] = 1 + math.log10(raw)

            # normalize document vector
            norm = math.sqrt(sum(w * w for w in self._tf[doc_id].values()))
            if norm > 0:
                for term in list(self._tf[doc_id].keys()):
                    self._tf[doc_id][term] /= norm

            # track highest document ID
            self._number_of_docs = max(self._number_of_docs, doc_id)

    def run_query(self, query):
        # tokenize and annotate negation in query
        tokens = query.lower().split()
        tokens = annotate_negation(tokens)
        return self._run_query(tokens)

    def _run_query(self, terms):
        # l:	logarithmic tf, t:	idf, n:	no normalizatio
        # build query tf
        query_tf = defaultdict(int)
        for t in terms:
            query_tf[t] += 1

        # compute query weights
        query_weight = {}
        for t, freq in query_tf.items():
            df = self._df.get(t, 0)
            if df > 0:
                idf = math.log10(self._number_of_docs / df)
                query_weight[t] = (1 + math.log10(freq)) * idf

        # score documents
        scores = defaultdict(float)
        for doc, vec in self._tf.items():
            for t in set(terms):
                if t in vec:
                    scores[doc] += vec[t] * query_weight.get(t, 0.0)

        # get top ten
        top = self.get_top_ten(scores)
        result = [doc for doc, _ in top]

        # backfill if fewer than needed
        for doc_id in range(1, self._number_of_docs + 1):
            if len(result) >= 1:
                break
            if doc_id not in result:
                result.append(doc_id)

        return result[0]

    def get_top_ten(self, scores):
        top_ten = []
        for doc, score in scores.items():
            if len(top_ten) < 10:
                top_ten.append((doc, score))
            else:
                # replace the smallest if current > smallest
                min_idx = min(range(len(top_ten)), key=lambda i: top_ten[i][1])
                if score > top_ten[min_idx][1]:
                    top_ten[min_idx] = (doc, score)
        top_ten.sort(key=lambda x: x[1], reverse=True)
        return top_ten
