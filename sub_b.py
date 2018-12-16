def split_train_test(data, target, split=0.5):
    n = int(len(data) * split)
    train_data = data[:n]
    train_target = target[:n]

    test_data = data[n:]
    test_target = target[n:]

    return train_data, train_target, test_data, test_target


def clean_data(data):
    import re

    REPLACE_NO_SPACE = re.compile("(\.)|(;)|(:)|(!)|(\')|(\?)|(,)|(\")|(\()|(\))|(\[)|(\])|(=)|(_)|(\*)|(://)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(-)|(/)(&\s\s\s)")

    def preprocess_reviews(text):
        text = [REPLACE_NO_SPACE.sub("", line.lower()) for line in text]
        text = [REPLACE_WITH_SPACE.sub(" ", line) for line in text]

        return text

    return preprocess_reviews(data)


def clean_bag_of_words_stop_words(data):
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stopwords_set = set(stopwords.words("english"))

    bag_of_words = []
    for text in data:
        words_filtered = [e.lower() for e in text.split() if len(e) >= 3]
        words_cleaned = [word for word in words_filtered
                         if 'http' not in word
                         and not word.startswith('@')
                         and not word.startswith('#')
                         and word != 'RT']
        words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
        bag_of_words.append(words_without_stopwords)

    return bag_of_words


def naive_bayes(train_data, train_target, test_data, test_target):
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

    guessed = 0
    for obj, target in zip(test_data, test_target):
        pred = classifier.classify(extract_features(obj.split()))
        if pred == target:
            guessed += 1

    print('[Accuracy]: %f = %s/%s ' % (guessed / len(test_data), guessed, len(test_data)))


def run_linear_regression(data, target, verbose=False):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(analyzer='word', lowercase=False, )
    features = cv.fit_transform(data)
    features_nd = features.toarray()  # for easy usage

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(features_nd, target, train_size=0.5, test_size=0.3,
                                                        random_state=1234, shuffle=False)

    from sklearn.linear_model import LogisticRegression
    log_model = LogisticRegression(solver='lbfgs')

    log_model = log_model.fit(X=X_train, y=y_train)

    y_pred = log_model.predict(X_test)

    from sklearn.metrics import accuracy_score
    print('[Accuracy]:', accuracy_score(y_test, y_pred))

    if verbose:
        feature_to_coef = {
            word: coef for word, coef in zip(
            cv.get_feature_names(), log_model.coef_[0]
        )
        }
        print("best positive words")
        for best_positive in sorted(
                feature_to_coef.items(),
                key=lambda x: x[1],
                reverse=True)[:5]:
            print(best_positive)

        print("best negative words")
        for best_negative in sorted(
                feature_to_coef.items(),
                key=lambda x: x[1])[:5]:
            print(best_negative)


def get_data():
    def get_data_from_file(filename, categories_d, r=32):
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
                answer = prts[2].strip()
                if cat == 'True' or cat == 'False':
                    data.append(answer)
                    target.append(categories_d[cat])

            return data, target

    categories_d = {'True': 0, 'False': 1}
    train_data, train_target = get_data_from_file('data/b/b-factual-q-a-clean-test.txt', categories_d)
    test_data, test_target = get_data_from_file('data/b/b-factual-q-a-clean-train.txt', categories_d)

    data = train_data + test_data
    target = train_target + test_target
    return data, target


def get_movie_data():
    def get_data_from_file(filename, category):
        with open(filename, 'r', encoding = "ISO-8859-1") as movies_file:

            data = movies_file.readlines()
            target = [category for _ in data]
            return data, target
    categories_d = {'objective': 0, 'subjective': 1}
    objective_data, objective_target = get_data_from_file('data/b/movie-objective.5000.txt', 0)
    subjective_data, subjective_target = get_data_from_file('data/b/movie-subjective.5000.txt', 1)
    data = objective_data + subjective_data
    target = objective_target + subjective_target

    indexing = list(range(len(data)))
    import random
    random.shuffle(indexing)
    shuffled_data = []
    shuffled_target = []
    for index in indexing:
        shuffled_data.append(data[index])
        shuffled_target.append(target[index])

    return shuffled_data, shuffled_target


def get_predictions_naive_bayes():
    data, target = get_data()
    data = clean_data(data)
    train_data, train_target, test_data, test_target = split_train_test(data, target)
    naive_bayes(train_data, train_target, test_data, test_target)


def get_predictions_linear_regression():
    data, target = get_data()
    data = clean_data(data)
    run_linear_regression(data, target)


def main():
    # TODO create different train test sets and compute average accuracy - these results change a lot based on split
    get_predictions_naive_bayes()
    # [Accuracy]: 0.647059 = 99/153

    get_predictions_linear_regression()
    # [Accuracy]: 0.6739130434782609


if __name__ == "__main__":
    main()
