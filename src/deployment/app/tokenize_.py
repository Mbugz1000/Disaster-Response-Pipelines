import re
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def tokenize(text):
    """
    In this method, we tokenize the text data. We take the following steps:
        1. Set the text to lower case
        2. Remove punctuation
        3. Tokenize the sentence
        4. Remove stop words
        5. Lematise the resulting tokens
    :param text: Input document
    :return: Cleaned tokens
    """
    text = text.lower()
    punctuation_removed = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    tokenized = word_tokenize(punctuation_removed)
    no_stop_words = [word for word in tokenized if word not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token).strip() for token in no_stop_words]

    return clean_tokens