from utils import get_data, plot_metrics, normalize
from model import MultiClassSVM, PCA
from typing import Tuple


def get_hyperparameters() -> Tuple[float, int, float]:
    # get the hyperparameters
    # C should be a list with median as 1 and 5 different values.
    raise NotImplementedError


def main() -> None:
    # hyperparameters
    learning_rate, num_iters, C = get_hyperparameters()

    # get data
    X_train, X_test, y_train, y_test = get_data()

    # normalize the data
    X_train, X_test = normalize(X_train, X_test)

    metrics = []
    for C_i in C :
        # reduce the dimensionality of the data
        pca = PCA(n_components=k)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # create a model
        svm = MultiClassSVM(num_classes=10)

        # fit the model
        svm.fit(
            X_train, y_train, C=C_i,
            learning_rate=learning_rate,
            num_iters=num_iters,
        )

        # evaluate the model
        accuracy = svm.accuracy_score(X_test, y_test)
        precision = svm.precision_score(X_test, y_test)
        recall = svm.recall_score(X_test, y_test)
        f1_score = svm.f1_score(X_test, y_test)

        metrics.append((k, accuracy, precision, recall, f1_score))

        print(f'k={k}, accuracy={accuracy}, precision={precision}, recall={recall}, f1_score={f1_score}')

    # plot and save the results
    plot_metrics(metrics)


if __name__ == '__main__':
    main()
