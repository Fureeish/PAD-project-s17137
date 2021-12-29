from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
x = data['data']
y = data['target']
names = data['target_names']
feature_names = data['feature_names']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True)

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

kmns = KMeans(init="k-means++", n_clusters=2, n_init=4, random_state=0)

kmns.fit(X_train, y_train)
out = kmns.fit_predict(X_test)

# Trzeba wyświetlać maksa z wyników oraz z odwrotności wyników
acc = accuracy_score(y_test, out) * 100
rec = recall_score(y_test, out) * 100
roc = roc_auc_score(y_test, out) * 100
print(f'\nAccuracy score : {max(acc, 100 - acc)}')
print(f'Recall score : {max(rec, 100 - rec)}')
print(f'ROC score : {max(roc, 100 - roc)}')
print(confusion_matrix(y_test, out))
