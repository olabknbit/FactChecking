def clean_bag_of_words_stop_words(data):
    import nltk
    from nltk.corpus import stopwords
    stopwords_set = set(stopwords.words("english"))
    bag_of_words = []
    for text in data:
        tokens = nltk.word_tokenize(text)
        words_filtered = [e.lower() for e in tokens if len(e) >= 3]
        words_cleaned = [word for word in words_filtered
                         if 'http' not in word
                         and not word.startswith('@')
                         and not word.startswith('#')
                         and word != 'RT']
        words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
        bag_of_words.append(words_without_stopwords)

    return bag_of_words


class NaiveBayes:
    def __init__(self):
        self.classifier = None
        self.extract_features = None

        # import nltk
        # nltk.download('stopwords')
        # nltk.download('punkt')

    def train(self, train_data, train_target):
        import nltk
        train_data_bag_of_words = clean_bag_of_words_stop_words(train_data)
        data = [(words_without_stopwords, label) for words_without_stopwords, label in
                zip(train_data_bag_of_words, train_target)]

        # Extracting word features
        def get_words_in_text(text):
            all = []
            for (words, sentiment) in text:
                all.extend(words)
            return all

        def get_word_features(wordlist):
            wordlist = nltk.FreqDist(wordlist)
            features = wordlist.keys()
            return features

        w_features = get_word_features(get_words_in_text(data))

        def extract_features(document):
            document_words = set(document)
            features = {}
            for word in w_features:
                features['contains(%s)' % word] = (word in document_words)
            return features

        # Training the Naive Bayes classifier
        training_set = nltk.classify.apply_features(extract_features, data)
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        self.classifier = classifier
        self.extract_features = extract_features
        return classifier, extract_features

    def get_predictions(self, test_data):
        if self.classifier is None:
            print("Error - please train first")
            exit(1)
        data = clean_bag_of_words_stop_words(test_data)
        predictions = [self.classifier.classify(self.extract_features(row)) for row in data]
        return predictions

    def run_naive_bayes_accuracy(self, test_data, test_target):
        predictions = self.get_predictions(test_data)
        from sklearn.metrics import accuracy_score
        return accuracy_score(test_target, predictions)
