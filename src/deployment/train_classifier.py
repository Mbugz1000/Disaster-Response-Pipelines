import sys
import re
import _pickle as cPickle
import warnings

import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV

warnings.filterwarnings("ignore")

home = '../../'
data_dir = home + '/data'
raw_dir = data_dir + '/raw'
processed_dir = data_dir + '/processed'
modelling_dir = data_dir + '/for_modelling'


def load_data(database_filepath):
    """
    This method loads the processed data from the saved DB

    :param database_filepath:
    :return: X, y and Column names of the target categories
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("select * from CategorisedMessages", engine)
    X = df['message']
    y = df.drop(columns=['message', 'id', 'original', 'genre'])

    return X, y, list(y.columns)


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


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        # 'vect__max_features': (None, 5000),
        'clf__estimator__n_estimators': [50, 100],
        # 'clf__estimator__min_samples_split': [2, 3, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    This method evaluates the performance of the model on all 35 categories. It then prints the median Accuracy,
    Precision and F1Score of the model.

    :param model:
    :param X_test:
    :param y_test:
    :param category_names:
    :return: None
    """
    y_pred = model.predict(X_test)

    rows = []
    for i in range(y_test.shape[1]):
        report = classification_report(y_test.values[:, i], y_pred[:, i], digits=3, output_dict=True)
        check_1 = report.get("1", None)
        output = dict(
            col_name=category_names[i],
            accuracy=report["accuracy"],
            precision=None if check_1 is None else report["1"]["precision"],
            specificity=report["0"]["recall"],
            sensitivity=None if check_1 is None else report["1"]["recall"],
            f1score=None if check_1 is None else report["1"]["f1-score"],
        )
        rows.append(output)

    performance_cv = pd.DataFrame(rows)

    print("Model Performance")
    output = dict(accuracy=performance_cv.accuracy.median(), precision=performance_cv.precision.median(),
                  f1score=performance_cv.f1score.median())
    print(output)


def save_model(model, model_filepath):
    """
    This method saves the model as a pickle file
    :param model:
    :param model_filepath:
    :return:
    """
    with open(model_filepath, mode='wb') as fp:
        cPickle.dump(model, fp)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()