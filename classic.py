from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC


def modelling(train_data, train_labels):
    model = SVC()
    model.fit(train_data, train_labels)
    return model


def evaluate(validation_data, validation_labels, models):
    y_pred = models.predict(validation_data)
    acc = accuracy_score(y_pred, validation_labels)
    clf_report = classification_report(y_pred, validation_labels)
    print(f'\nAccuracy score:\n{acc}\n')
    print(f'Classification report:\n{clf_report}')