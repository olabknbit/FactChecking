def prep_data(data):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(analyzer='word', lowercase=False, stop_words='english')
    return cv.fit_transform(data).toarray()


class NaiveBayes:
    def __init__(self, serial_filename=None):
        from sklearn.naive_bayes import GaussianNB
        self.classifier = GaussianNB()
        self.serial_filename = serial_filename

    def train(self, train_data, train_target):
        # Training the Naive Bayes classifier
        self.classifier.fit(train_data, train_target)
        self.serialize()
        return self.classifier

    def get_predictions(self, test_data):
        if self.classifier is None:
            print("Error - please train first")
            exit(1)
        return self.classifier.predict(test_data)

    def run_naive_bayes_accuracy(self, test_data, test_target):
        predictions = self.get_predictions(test_data)
        from sklearn.metrics import accuracy_score
        return accuracy_score(test_target, predictions)

    def serialize(self):
        import pickle
        s = pickle.dumps(self.classifier)
        with open(self.serial_filename, 'wb') as f:
            f.write(s)
