import os

import numpy as np
import pandas as pd
import scipy.stats
import sklearn
import sklearn_crfsuite
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn_crfsuite import metrics, scorers
from tqdm import tqdm


def make_df_for_sent(single_sent):
    """Make a DataFrame out of all the records for a single sentence."""
    df = pd.DataFrame(data=single_sent, columns=['word', 'pos', 'parse', 'ner'])
    df.index.name = 'word_seq_num'
    return df
    
def all_sentences(sents):
    """Convert the list of list of lists to a list of DataFrames."""
    total_df = [make_df_for_sent(s) for s in sents]
    return total_df

def get_labels(all_sents):
    """Return the labels for all the words in a collection of sentences."""
    all_labels = []
    
    for s_df in all_sents:
        labels = s_df.loc[:, 'ner'].tolist()
        all_labels.append(labels)
        
    return all_labels 


def word2features(i, single_sent_df):
    """
    Return a dictionary of feature names and values for the word at ``word_idx`` 
    in a single sentence represented as a ``DataFrame``."""
    
    word, postag = single_sent_df.iloc[i].loc[['word', 'pos']]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1, postag1 = single_sent_df.iloc[i-1].loc[['word', 'pos']]
        
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < (single_sent_df.shape[0] - 1):
        word1, postag1 = single_sent_df.iloc[i+1].loc[['word', 'pos']]
        
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(s_df):
    """
    Return the feature values extracted from a single sentence.
    """    
    features = s_df.index.map(lambda word_idx: word2features(word_idx, s_df))
    return features.tolist()

def get_feature_values(all_sents):
    """Get the feature values for all the sentences in train/test dataset."""
    
    all_features = [sent2features(s) for s in tqdm(all_sents)]    
    return all_features

def calculate_metrics_crf(y_true, y_pred, labels):
    y_test_flat = [item for sublist in y_true for item in sublist]
    y_pred_flat = [item for sublist in y_pred for item in sublist]

    # group B and I results
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )

    report = classification_report(y_test_flat, y_pred_flat,
                                   labels=sorted_labels,
                                   digits=3)
    
    return report