def prep_data(data):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(analyzer='word', lowercase=False, stop_words='english')
    return cv.fit_transform(data).toarray()


class NaiveBayes:
    def __init__(self, onnx_filename=None):
        from sklearn.naive_bayes import GaussianNB
        self.classifier = GaussianNB()
        # from sklearn.naive_bayes import MultinomialNB
        # self.classifier = MultinomialNB()
        self.onnx_filename = onnx_filename
        self.onnx_classifier = None

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
        # Convert into ONNX format with onnxmltools
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import Int64TensorType
        initial_type = [('int64_input', Int64TensorType([1, 2365]))]
        self.onnx_classifier = convert_sklearn(self.classifier, initial_types=initial_type)
        with open(self.onnx_filename, "wb") as f:
            f.write(self.onnx_classifier.SerializeToString())
