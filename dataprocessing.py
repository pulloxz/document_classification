from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

#dataset directories
dataset_directories = []

#list to store processed data from all datasets
all_document_texts = []
custom_stopwords_file = 'stopword'

# Reading custom stopwords once
with open(custom_stopwords_file, 'r') as stopword_file:
    custom_stopwords = set(stopword_file.read().splitlines())

# Create a TfidfVectorizer
tfidf = TfidfVectorizer(lowercase=True)

# The remove_custom_stopwords function
def remove_custom_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in custom_stopwords]
    return ' '.join(filtered_words)

# Process and accumulate data from all datasets
for data_directory in dataset_directories:
    document_texts = []

    for filename in os.listdir(data_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                document_texts.append(text)

    # Preprocess the text data
    document_texts = [remove_custom_stopwords(text) for text in document_texts]

    # Append the preprocessed text data to the list
    all_document_texts.extend(document_texts)

# Create TF-IDF features for the accumulated text data
tfidf_result = tfidf.fit_transform(all_document_texts)

# Save the tfidf_result to a file using joblib
joblib.dump(tfidf_result, 'tfidf_result.pkl')

# Print the vocabulary, if needed
print('\nindexes numbers:')
print(tfidf.vocabulary_)

# Print the TF-IDF values, if needed
print('\nTF-IDF values:')
print(tfidf_result)
