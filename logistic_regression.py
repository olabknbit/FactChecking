def prep_data(data):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(analyzer='word', lowercase=False, )
    return cv.fit_transform(data).toarray()


class LogisticRegression:
    def __init__(self, onnx_filename=None):
        from sklearn.linear_model import LogisticRegression
        self.log_model = LogisticRegression(solver='lbfgs')
        self.onnx_filename = onnx_filename

    def train(self, X_train, y_train):
        self.log_model.fit(X=X_train, y=y_train)
        self.serialize()
        return self.log_model

    def get_predictions(self, X_test):
        return self.log_model.predict(X_test)

    def get_accuracy(self, data, y_test):
        y_pred = self.get_predictions(data)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_test, y_pred)

    def serialize(self):
        pass
