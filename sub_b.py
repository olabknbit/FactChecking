def get_data(filename, categories_d, r=32):
    with open(filename, 'r') as questions_file:
        data = []
        target = []
        lines = questions_file.readlines()

        import random
        random.seed(r)
        random.shuffle(lines)

        for line in lines:
            prts = line.split('\t')
            cat = prts[1]
            question = prts[2].strip()
            if cat == 'True' or cat == 'False':
                data.append(question)
                target.append(categories_d[cat])

        return data, target


def split_train_test(data, target, split=0.6):
    n = int(len(data) * split)
    train_data = data[:n]
    train_target = target[:n]

    test_data = data[n:]
    test_target = target[n:]

    return train_data, train_target, test_data, test_target


def main():
    categories_d = {'True': 0, 'False': 1}

    train_data, train_target = get_data('data/b/b-factual-q-a-clean-train.txt', categories_d)
    test_data, test_target = get_data('data/b/b-factual-q-a-clean-test.txt', categories_d)

    data = train_data + test_data
    target = train_target + test_target

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(analyzer='word', lowercase=False, )
    features = vectorizer.fit_transform(data)
    features_nd = features.toarray()  # for easy usage

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(features_nd, target, train_size=0.7, test_size=0.3,
                                                        random_state=1234)

    from sklearn.linear_model import LogisticRegression
    log_model = LogisticRegression(solver='lbfgs')

    log_model = log_model.fit(X=X_train, y=y_train)

    y_pred = log_model.predict(X_test)

    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
