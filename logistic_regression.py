def prep_data(data):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(analyzer='word', lowercase=False, )
    features = cv.fit_transform(data)
    return features.toarray()  # for easy usage


class LogisticRegression:
    def __init__(self):
        self.log_model = None

    def train(self, X_train, y_train):
        from sklearn.linear_model import LogisticRegression
        log_model = LogisticRegression(solver='lbfgs')

        self.log_model = log_model.fit(X=X_train, y=y_train)
        return self.log_model

    def get_predictions(self, X_test):
        return self.log_model.predict(X_test)

    def get_accuracy(self, data, y_test):
        y_pred = self.get_predictions(data)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_test, y_pred)
