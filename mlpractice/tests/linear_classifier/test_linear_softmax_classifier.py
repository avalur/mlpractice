from mlpractice.utils import ExceptionInterception

try:
    from mlpractice_solutions.mlpractice_solutions\
        .linear_classifier_solution import LinearSoftmaxClassifier
except ImportError:
    LinearSoftmaxClassifier = None

from data_prep import load_and_split_data


# pre-train tests
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


def test_all(linear_softmax_classifier=LinearSoftmaxClassifier):
    data = load_and_split_data()
    test_output_shape(data, linear_softmax_classifier)

    print("All tests passed!")
