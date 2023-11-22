from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

tfidf_result = joblib.load('tfidf_result.pkl')

labels = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100 + [5] * 100 + [6] * 100 + [7] * 100 + [8] * 100 + [9] * 100)

X_train, X_test, y_train, y_test = train_test_split(tfidf_result, labels, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', C=1.0)

svm_model.fit(X_train, y_train)

joblib.dump(svm_model, 'svm_model.pkl')

y_pred = svm_model.predict(X_test)

vectorizer = joblib.load('tfidf_vectorizer.pkl')
vocabulary_used_during_training = joblib.load('tfidf_vocabulary.pkl')

print('\nClassification Report:')
print(classification_report(y_test, y_pred))

print('\nAccuracy:', accuracy_score(y_test, y_pred))
