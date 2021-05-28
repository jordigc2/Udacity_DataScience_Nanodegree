import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import re
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("messages", engine)
    df = df.drop(labels=['related'], axis=1)
    X = df.message.values
    Y = df.iloc[:, 4:].values
    column_names = df.iloc[:, 4:].columns

    return X,Y, column_names


def tokenize(text):
    
    #replacing the url by urlplaceholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    #Remove punctuation and convert to lower case
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text).lower()
    
    #split the text in a list of words, called tokens
    tokens = nltk.word_tokenize(text)
    
    #
    lemmatizer = nltk.WordNetLemmatizer()

    clean_tokens = []
    
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]

    return clean_tokens


def build_model(algo='rfc'):

    def get_accuracy_score(y_test, y_pred):
        
        scores = []
        
        for indx in range(y_test.shape[-1]):
            scores.append( accuracy_score(y_test[:, indx], y_pred[:, indx]))
            
        return np.mean(scores)

    accuracy_scorer = make_scorer(get_accuracy_score)
    if algo == 'rfc':
        pipeline_rfc = Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer()),
                ('classifier', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
        ])
        parameters = {'classifier__estimator__max_depth': [50, 75],  
                      'classifier__estimator__n_estimators': [100, 250]}
        model = GridSearchCV(
                                pipeline_rfc, 
                                param_grid=parameters, 
                                scoring=accuracy_scorer, 
                                verbose=10, 
                                return_train_score=True,
                                n_jobs=-1
                                )

    elif algo == 'ada':
        pipeline_ada = Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer()),
                ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
        ])
        parameters_grid = {'classifier__estimator__learning_rate': [0.02, 0.05],
                           'classifier__estimator__n_estimators': [20, 40]}
        model = GridSearchCV(
                            pipeline_ada, 
                            param_grid=parameters_grid, 
                            scoring=accuracy_scorer, 
                            n_jobs=-1
                            )
    return model


def evaluate_model(model, X_test, y_test, category_names):

    y_pred = model.predict(X_test)
    scores = pd.DataFrame()
    for indx in range(y_test.shape[-1]):
        
        accuracy = accuracy_score(y_test[:, indx], y_pred[:, indx])
        precision =round( precision_score(y_test[:, indx], y_pred[:, indx], average='micro'))
        recall = recall_score(y_test[:, indx], y_pred[:, indx], average='micro')
        f_1 = f1_score(y_test[:, indx], y_pred[:, indx], average='micro')
        
        score = {
        'Category':category_names[indx],
        'Accuracy': accuracy, 
        'Precision': precision, 
        'Recall': recall, 
        'F1 Score': f_1}
        
        scores = scores.append(score, ignore_index=True)
    scores = scores.set_index('Category')
    print("Score: \n", scores)
    print("==================================")
    print(("The average accuracy score among all categories is {:.4f},\nthe average precision score score among all categories is {:.4f},\nthe average recall score among all categories is {:.4f},\nthe average F 1 score among all categories is {:.4f}").
          format(scores.mean()["Accuracy"],scores.mean()["Precision"],
                 scores.mean()["Recall"],scores.mean()["F1 Score"]))
    return scores


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(algo='rfc')
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()