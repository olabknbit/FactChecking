def prep_data(data):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(analyzer='word', lowercase=False, )
    return cv.fit_transform(data).toarray()


class LogisticRegression:
    def __init__(self, serial_filename=None):
        from sklearn.linear_model import LogisticRegression
        self.classifier = LogisticRegression(solver='lbfgs')
        self.serial_filename = serial_filename

    def train(self, X_train, y_train):
        self.classifier.fit(X=X_train, y=y_train)
        self.serialize()
        return self.classifier

    def get_predictions(self, X_test):
        return self.classifier.predict(X_test)

    def get_accuracy(self, data, y_test):
        y_pred = self.get_predictions(data)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_test, y_pred)

    def serialize(self):
        import pickle
        s = pickle.dumps(self.classifier)
        with open(self.serial_filename, 'wb') as f:
            f.write(s)
