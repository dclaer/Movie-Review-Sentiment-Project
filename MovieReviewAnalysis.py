"""
Predicting Movie Review Sentiment
"""

import argparse
import os
import re
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# plotting
import matplotlib.pyplot as plt

# nltk - optional parts
try:
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    nltk_available = True
except Exception:
    nltk_available = False

# Utility: basic text cleaning and token processing
def clean_text(text):
    """Lowercase, remove symbols, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_normalize(text, use_lemmatize=True):
    """Tokenize and optionally lemmatize/stem. Returns joined string (for vectorizers)."""
    text = clean_text(text)
    tokens = text.split()
    # remove stopwords if available
    if nltk_available:
        try:
            stops = set(stopwords.words('english'))
            tokens = [t for t in tokens if t not in stops]
        except LookupError:
            # download on demand
            nltk.download('stopwords')
            stops = set(stopwords.words('english'))
            tokens = [t for t in tokens if t not in stops]
    # lemmatize if possible, else stem
    if nltk_available and use_lemmatize:
        try:
            wn = WordNetLemmatizer()
            # ensure wordnet downloaded
            _ = nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            wn = WordNetLemmatizer()
        tokens = [wn.lemmatize(t) for t in tokens]
    else:
        # fallback stemmer
        if nltk_available:
            ps = PorterStemmer()
            tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)

def load_imdb_from_csv(path, text_col='review', label_col='sentiment', limit=None):
    """Download from Kaggle: sentiment column usually 'positive'/'negative' strings - map to 1/0"""
    df = pd.read_csv(path)
    if limit:
        df = df.sample(limit, random_state=42)
    # try to be flexible with column names
    if label_col not in df.columns:
        # try variations
        possible_labels = [c for c in df.columns if 'sentiment' in c.lower() or 'label' in c.lower()]
        if possible_labels:
            label_col = possible_labels[0]
        else:
            raise ValueError('Label column not found in CSV. Columns: {}'.format(df.columns.tolist()))
    if text_col not in df.columns:
        possible_text = [c for c in df.columns if 'review' in c.lower() or 'text' in c.lower()]
        if possible_text:
            text_col = possible_text[0]
        else:
            raise ValueError('Text column not found in CSV. Columns: {}'.format(df.columns.tolist()))
    # map label strings to 0/1 if necessary
    if df[label_col].dtype == object:
        df[label_col] = df[label_col].str.lower().map({'positive':1, 'negative':0})
    X = df[text_col].astype(str).tolist()
    y = df[label_col].astype(int).tolist()
    return train_test_split(X, y, test_size=0.5, random_state=42)  # mimic TFDS split sizes

def build_pipeline(vectorizer='tfidf', model='nb', max_features=20000):
    """Return a sklearn Pipeline with preprocessing vectorizer and chosen model."""
    if vectorizer == 'count':
        vec = CountVectorizer(max_features=max_features)
    else:
        vec = TfidfVectorizer(max_features=max_features)
    if model == 'nb':
        clf = MultinomialNB()
    elif model == 'logreg':
        clf = LogisticRegression(max_iter=1000)
    elif model == 'svm':
        clf = LinearSVC(max_iter=10000)
    else:
        raise ValueError('Unknown model: {}'.format(model))
    pipe = Pipeline([('vect', vec), ('clf', clf)])
    return pipe

def evaluate_model(pipe, X_train, X_test, y_train, y_test, save_prefix=None):
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, digits=4, output_dict=False)
    cm = confusion_matrix(y_test, preds)
    results = dict(accuracy=float(acc), report=report, confusion_matrix=cm.tolist())
    if save_prefix:
        joblib.dump(pipe, f"{save_prefix}_model.joblib")
        with open(f"{save_prefix}_metrics.json", 'w') as f:
            json.dump(results, f, indent=2)
    return results, preds, cm

def plot_confusion(cm, title='Confusion Matrix', outpath=None):
    plt.figure(figsize=(4,3))
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha='center', va='center')
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    plt.close()

def main(args=None):
    # load data
    if args.demo:
        # tiny synthetic dataset for a quick demo
        X = [
            "The movie was amazing and fun",
            "I hated the movie, it was boring and too long",
            "Fantastic acting and great plot",
            "Terrible movie, waste of time",
            "I loved the characters and the story",
            "Bad directing and poor pacing",
            "A delightful and charming film",
            "Not good, the acting was awful",
        ]
        y = [1,0,1,0,1,0,1,0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    elif args.use_tfds:
        print("Loading IMDb using tensorflow_datasets (this may download ~80MB)...")
        X_train, y_train, X_test, y_test = load_imdb_with_tfds(limit=args.limit)
    elif args.csv:
        X_train, X_test, y_train, y_test = load_imdb_from_csv(args.csv, limit=args.limit)
    else:
        raise RuntimeError('No data source specified. Use --demo, --use-tfds or --csv path.')

    # preprocessing on text columns (apply tokenization/lemmatization optionally)
    if args.preprocess:
        print('Preprocessing text (tokenize, remove stopwords, lemmatize/stem where available)...')
        X_train = [tokenize_and_normalize(t, use_lemmatize=not args.use_stem) for t in X_train]
        X_test = [tokenize_and_normalize(t, use_lemmatize=not args.use_stem) for t in X_test]

    # build pipeline and evaluate
    pipe = build_pipeline(vectorizer=args.vectorizer, model=args.model, max_features=args.max_features)
    print('Training model:', args.model, 'with', args.vectorizer)
    results, preds, cm = evaluate_model(pipe, X_train, X_test, y_train, y_test, save_prefix=args.outprefix if args.save else None)
    print('Accuracy:', results['accuracy'])
    print('Classification report:')
    print(results['report'])
    print('Confusion matrix:')
    print(np.array(results['confusion_matrix']))

    # plot confusion if requested
    if args.save and args.outprefix:
        plot_confusion(np.array(results['confusion_matrix']), title=f"CM_{args.model}_{args.vectorizer}", outpath=f"{args.outprefix}_cm.png")
        print('Saved model and metrics to', args.outprefix + '_model.joblib and ' + args.outprefix + '_metrics.json')

    # optionally run GridSearch for hyperparameter tuning (small example)
    if args.gridsearch and not args.demo:
        print('Running small grid search (this can be slow)...')
        param_grid = {
            'vect__max_features': [5000, 20000],
            'clf__C' if args.model in ('logreg','svm') else 'clf__alpha': [0.1, 1.0]
        }
        # for NaiveBayes key is clf__alpha, for logreg/svm it's clf__C - handle gracefully
        pipe_for_gs = build_pipeline(vectorizer=args.vectorizer, model=args.model, max_features=args.max_features)
        # create a compatible param_grid by checking which param exists
        real_param_grid = {}
        if 'clf__C' in param_grid:
            real_param_grid = {'vect__max_features': param_grid['vect__max_features'], 'clf__C': param_grid['clf__C']}
        else:
            real_param_grid = {'vect__max_features': param_grid['vect__max_features'], 'clf__alpha': param_grid['clf__alpha']}
        gs = GridSearchCV(pipe_for_gs, real_param_grid, cv=3, n_jobs=1, verbose=1)
        gs.fit(X_train, y_train)
        print('Best params:', gs.best_params_)
        best = gs.best_estimator_
        preds_gs = best.predict(X_test)
        print('GridSearch accuracy:', accuracy_score(y_test, preds_gs))

if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--use-tfds", action="store_true")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--use-stem", action="store_true")
    parser.add_argument("--vectorizer", type=str, default="tfidf")
    parser.add_argument("--model", type=str, default="svm")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--outprefix", type=str, default="run")
    parser.add_argument("--gridsearch", action="store_true")

    if len(sys.argv) == 1:
        sys.argv.extend(["--demo", "--preprocess"])

    args = parser.parse_args()

    main(args)



