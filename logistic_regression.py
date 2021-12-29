from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
x = data['data']
y = data['target']
names = data['target_names']
feature_names = data['feature_names']

# x_train, y_train = x[:, :2], y
x_train, y_train = x[:, :2], y

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, shuffle=True)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f'\nAccuracy score : {accuracy_score(y_test, y_pred) * 100}')
print(f'Recall score : {recall_score(y_test, y_pred) * 100}')
print(f'ROC score : {roc_auc_score(y_test, y_pred) * 100}')
print(confusion_matrix(y_test, y_pred))
