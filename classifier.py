import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import spacy as sp
import pickle
import re
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore tensorflow warnings

def clean_text(title, abstract):
    clean_text = title + ' ' + abstract
    
    # remove punctuation marks
    punctuation ='\!\"\#\$\%\&\(\)\*\+\-\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~'
    clean_text = re.sub(punctuation, '', clean_text)

    # convert text to lowercase
    clean_text = clean_text.lower()

    # remove numbers
    clean_text = clean_text.replace("[0-9]", "")

    # remove trailing whitespaces
    clean_text = ' '.join(clean_text.split())
    
    # lemmatise
    nlp = sp.load('en', disable=['parser', 'ner'])
    clean_text = ' '.join(np.asarray([token.lemma_ 
        for token in nlp(clean_text)]))
 
    return clean_text

def elmo_clean_text(clean_text):
    # load the ELMo model
    elmo = hub.Module("elmo", trainable = False)

    with tf.compat.v1.Session() as sess:
        # initialise tensorflow session
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        
        # extract ELMo features
        embeddings = elmo([clean_text], signature = "default", 
            as_dict = True)["elmo"]

        # save the average ELMo features for every title+abstract
        elmo_vectors = sess.run(tf.reduce_mean(embeddings, 1))

    return elmo_vectors

def predict_cats_from_elmo(elmo_vectors, nn_model):
    pred_cats = np.around(nn_model.predict(elmo_vectors), decimals = 0)
    pred_cats = np.squeeze(pred_cats).T
    pred_cats = pred_cats.astype('int')
    return pred_cats

def predict_cats(title, abstract, model_path = 'nn_model.pickle'):
    elmo_vectors = elmo_clean_text(clean_text(title, abstract))
    
    # load neural network model
    with open(model_path, 'rb') as pickle_in:
        nn_model = pickle.load(pickle_in)

    pred_cat_probs = predict_cats_from_elmo(elmo_vectors.T, nn_model)
    return pred_cat_probs > 0.5
