import joblib

from dataprocessing import remove_custom_stopwords

svm_model = joblib.load('svm_model.pkl')

tfidf = joblib.load('tfidf_vectorizer.pkl')

vocabulary = joblib.load('tfidf_vocabulary.pkl')

tfidf.vocabulary_ = vocabulary
label_to_class = {0: 'sport', 1: 'business', 2: 'entertainment', 3: 'food', 4: 'technology', 5: 'space', 6: 'politics', 7: 'medical', 8: 'historical', 9: 'graphics'}

new_file_path = ''  # Replace with the path to your new file
with open(new_file_path, 'r', encoding='utf-8') as file:
    new_text = file.read()

new_text = remove_custom_stopwords(new_text)  # Assuming remove_custom_stopwords function is available

new_text_tfidf = tfidf.transform([new_text])

predicted_label = svm_model.predict(new_text_tfidf)
predicted_class = label_to_class[predicted_label[0]]

print('Predicted Label:', predicted_class)
