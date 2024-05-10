# Import necessary libraries
import sys
import numpy as np
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    """
    Load data from SQLite database.

    Parameters:
    database_filepath (str): Filepath of the SQLite database.

    Returns:
    X (DataFrame): Features dataframe.
    Y (DataFrame): Target dataframe.
    category_names (list): List of category names.
    """
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)
    
    # Define features and target variables
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize text data.

    Parameters:
    text (str): Text data to be tokenized.

    Returns:
    clean_tokens (list): List of clean tokens.
    """
    # Remove URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Lemmatization and remove stopwords
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word not in stop_words]
    
    return clean_tokens

def build_model():
    """
    Build machine learning pipeline.

    Returns:
    model (GridSearchCV): Machine learning model.
    """
    # Build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=0)))
    ])

    # Define parameters for grid search
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [3]
    }

    # Grid search
    model = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=1)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance.

    Parameters:
    model: Trained machine learning model.
    X_test (DataFrame): Testing features.
    Y_test (DataFrame): Testing targets.
    category_names (list): List of category names.
    """
    # Predict on test data
    Y_pred = model.predict(X_test)
    
    # Print classification report for each category
    for i, category in enumerate(category_names):
        print(f"Category: {category}\n")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        print("-----------------------------------------------------\n")

def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.

    Parameters:
    model: Trained machine learning model.
    model_filepath (str): Filepath to save the model.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    # Check for correct number of arguments
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1], sys.argv[2]
        print(f"Loading data from {database_filepath}...\n")
        
        # Load data
        X, Y, category_names = load_data(database_filepath)
        
        # Split data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=0)
        
        print("Building model...\n")
        
        # Build model
        model = build_model()
        
        print("Training model...\n")
        
        # Train model
        model.fit(X_train, Y_train)
        
        print("Evaluating model...\n")
        
        # Evaluate model
        evaluate_model(model, X_test, Y_test, category_names)
        
        print(f"Saving model to {model_filepath}...\n")
        
        # Save model
        save_model(model, model_filepath)
        
        print("Model saved successfully!")
    
    else:
        print("Please provide the filepath of the disaster messages database "\
              "and the filepath to save the model.\n"\
              "For example: python train_classifier.py ../data/DisasterResponse.db classifier.pkl")

if __name__ == '__main__':
    main()
