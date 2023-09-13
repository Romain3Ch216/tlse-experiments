import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


def knn_pred(X_train, Y_train, X_test, Y_test):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)

    metrics = {
        'OA': accuracy_score(pred, Y_test),
        'F1': list(f1_score(pred, Y_test, average=None)),
        'avg F1': f1_score(pred, Y_test, average='macro')
    }

    return metrics
