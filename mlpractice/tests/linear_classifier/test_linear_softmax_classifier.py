from mlpractice.utils import ExceptionInterception

try:
    from mlpractice_solutions.mlpractice_solutions\
        .linear_classifier_solution import LinearSoftmaxClassifier
except ImportError:
    LinearSoftmaxClassifier = None

from sklearn.metrics import accuracy_score
from data_prep import load_and_split_data


def test_output_shape(data, linear_softmax_classifier=LinearSoftmaxClassifier):
    with ExceptionInterception():
        X_train, X_test, y_train, y_test = data
        clf = linear_softmax_classifier()
        clf.fit(X_train, y_train)
        train_predict = clf.predict(X_train)
        test_predict = clf.predict(X_test)

        assert train_predict.shape == (X_train.shape[0],), \
            "Classifier output must match the number of input samples"
        assert test_predict.shape == (X_test.shape[0],), \
            "Classifier output must match the number of input samples"


def test_increase_acc(data, linear_softmax_classifier=LinearSoftmaxClassifier):
    with ExceptionInterception():
        X_train, _, y_train, _ = data
        acc_list = []
        for num_epochs in [1, 10, 50, 100, 200, 300, 400]:
            clf = linear_softmax_classifier()
            clf.fit(X_train, y_train, batch_size=20, epochs=num_epochs)
            y_pred_train = clf.predict(X_train)
            acc_list.append(accuracy_score(y_train, y_pred_train))

        assert sorted(acc_list) == acc_list, \
            "Accuracy should increase as number of epochs increase(until some moment)"


def test_evaluation(data, linear_softmax_classifier=LinearSoftmaxClassifier):
    with ExceptionInterception():
        X_train, X_test, y_train, y_test = data
        clf = linear_softmax_classifier()
        clf.fit(X_train, y_train, batch_size=20, epochs=400)
        y_pred = clf.predict(X_test)

        assert accuracy_score(y_test, y_pred) > 0.94, \
            "Accuracy on test should be > 0.94"


def test_all(linear_softmax_classifier=LinearSoftmaxClassifier):
    data = load_and_split_data()
    data_scaled = load_and_split_data(scale=True)

    # pre-train tests
    test_output_shape(data, linear_softmax_classifier)
    test_increase_acc(data_scaled, linear_softmax_classifier)
    # evaluation tests
    test_evaluation(data_scaled, linear_softmax_classifier)

    print("All tests passed!")
