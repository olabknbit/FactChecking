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
            question = prts[2]
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
    categories_d = {'socializing': 0, 'factual': 1, 'opinion': 2}
    categories = ['socializing', 'factual', 'opinion']

    data, target = get_data('data/all-yquestions-clean.txt', categories_d)
    train_data, train_target, test_data, test_target = split_train_test(data, target)

    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data)

    from sklearn.feature_extraction.text import TfidfTransformer
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    tf_transformer.transform(X_train_counts)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB().fit(X_train_tfidf, train_target)

    X_new_counts = count_vect.transform(test_data)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    n = len(test_data)
    guessed = 0
    for doc, category_predicted, category_real in zip(test_data, predicted, test_target):
        if category_predicted == category_real:
            guessed += 1
        else:
            print('%r => %s | %s' % (doc, categories[int(category_predicted)], categories[int(category_real)]))
    print('accuracy', guessed / n)


if __name__ == "__main__":
    main()
