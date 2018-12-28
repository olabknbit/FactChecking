def get_vectorizer():
    import nltk, string
    from sklearn.feature_extraction.text import TfidfVectorizer

    stemmer = nltk.stem.porter.PorterStemmer()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

    def stem_tokens(tokens):
        return [stemmer.stem(item) for item in tokens]

    '''remove punctuation, lowercase, stem'''

    def normalize(text):
        return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

    return TfidfVectorizer(tokenizer=normalize, stop_words='english')


class CosineSimilarity:
    def __init__(self):
        self.vectorizer = get_vectorizer()
        import nltk
        nltk.download('punkt')

    def cosine_similarity(self, text1, text2):
        tfidf = self.vectorizer.fit_transform([text1, text2])
        return ((tfidf * tfidf.T).A)[0, 1]

    def predict(self, data):
        snippets, answers, tags = data.get_q_query_snippet_answer_pairs()
        similarities = []
        for snippet, answer, tag in zip(snippets, answers, tags):
            similarities.append(self.cosine_similarity(snippet, answer))

        return similarities, tags
