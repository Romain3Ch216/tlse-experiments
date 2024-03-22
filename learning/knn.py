import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


def knn_pred(X_train, Y_train, X_test, Y_test):
    clf = KNeighborsClassifier(n_neighbors=3)
    Y_train = Y_train.reshape(-1)
    Y_test = Y_test.reshape(-1)
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_test.shape[0] == Y_test.shape[0]
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)

    metrics = {
        'OA': accuracy_score(pred, Y_test),
        'F1': list(f1_score(pred, Y_test, average=None)),
        'avg F1': f1_score(pred, Y_test, average='macro')
    }

    return metrics
