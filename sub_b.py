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
        with open(filename, 'r', encoding="ISO-8859-1") as movies_file:
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


def get_predictions_naive_bayes(train_data, train_target, test_data):
    from naive_bayes import NaiveBayes
    nb = NaiveBayes()
    nb.train(train_data, train_target)
    return nb.get_predictions(test_data)


def get_accuracy_naive_bayes(train_data, train_target, test_data, test_target):
    predictions = get_predictions_naive_bayes(train_data, train_target, test_data)
    from sklearn.metrics import accuracy_score
    return accuracy_score(test_target, predictions)


def get_predictions_logistic_regression(train_data, train_target, test_data):
    from logistic_regression import LogisticRegression
    lr = LogisticRegression()
    lr.train(train_data, train_target)
    return lr.get_predictions(test_data)


def get_accuracy_logistic_regression(train_data, train_target, test_data, test_target):
    predictions = get_predictions_logistic_regression(train_data, train_target, test_data)
    from sklearn.metrics import accuracy_score
    return accuracy_score(test_target, predictions)


def run_cross_validation(data, target):
    import numpy as np
    from logistic_regression import prep_data
    lr_data = np.array(prep_data(data))

    nb_data = np.array(data)

    positives_accuracy_nb, negatives_accuracy_nb, positives_accuracy_lr, negatives_accuracy_lr = 0., 0., 0., 0.
    negatives_count, positives_count = 0, 0
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    loo.get_n_splits(data)

    target = np.array(target)
    nb_preds = []
    lr_preds = []
    for train_index, test_index in loo.split(data):
        train_data, test_data = nb_data[train_index], nb_data[test_index]
        train_target, test_target = target[train_index], target[test_index]
        nb_pred = get_predictions_naive_bayes(train_data, train_target, test_data)

        train_data, test_data = lr_data[train_index], lr_data[test_index]
        lr_pred = get_predictions_logistic_regression(train_data, train_target, test_data)
        if test_target[0] == 0:
            negatives_count += 1.
            if nb_pred[0] == 0:
                negatives_accuracy_nb += 1.
            if lr_pred[0] == 0:
                negatives_accuracy_lr += 1.
        else:
            positives_count += 1.
            if nb_pred[0] == 1:
                positives_accuracy_nb += 1.
            if lr_pred[0] == 1:
                positives_accuracy_lr += 1.
        nb_preds.append(nb_pred[0])
        lr_preds.append(lr_pred[0])

    if positives_count == 0:
        positives_count = 1
    if negatives_count == 0:
        negatives_count = 1

    print('NB p %f (%d/%d) NB n %f (%d/%d) \nLR p %f  (%d/%d) LR n %f (%d/%d) '
          % (positives_accuracy_nb / positives_count, positives_accuracy_nb, positives_count,
             negatives_accuracy_nb / negatives_count, negatives_accuracy_nb, negatives_count,
             positives_accuracy_lr / positives_count, positives_accuracy_lr, positives_count,
             negatives_accuracy_lr / negatives_count, negatives_accuracy_lr, negatives_count))
    return nb_preds, lr_preds, target


def store_preds(preds, title):
    with open('data/b/' + title + '.txt', 'w') as f:
        f.writelines([str(pred) + '\n' for pred in preds])


def main():
    data, target = get_data()

    ## get_accuracy_naive_bayes(data, target)
    # Accuracy %d 0.6557377049180327

    ## get_accuracy_logistic_regression(data, target):
    # from logistic_regression import prep_data
    # features_nd = prep_data(data)
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(features_nd, target, train_size=0.5, test_size=0.3,
    #                                                     random_state=1234, shuffle=False)
    # # print(y_test[0], X_test[0])
    # acc=get_accuracy_logistic_regression(X_train, y_train, X_test, y_test)
    # print(acc)
    # [Accuracy]: 0.6739130434782609

    nb_preds, lr_preds, target = run_cross_validation(data, target)
    store_preds(nb_preds, "nb_preds")
    store_preds(lr_preds, "lr_preds")


if __name__ == "__main__":
    main()
