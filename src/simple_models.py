def load_data(fname: str = 'arxiv_data_pp', data_dir: str = '.data',
    split_ratio: float = 0.98, random_state: int = 42, 
    use_fasttext: bool = False):
    ''' Load data in one big chunk.
    
    INPUT
        fname: str = 'arxiv_data_pp'
            The name of the tsv file containing the data, without file
            extension
        split_ratio: float = 0.98
            How large the training set will be in proportion to all the data
        use_fasttext: bool = False
            Whether to use the homemade FastText vectors. Otherwise bag
            of words vectors will be used
        random_state: int = 42
            Random state for reproducibility
        data_dir: str = '.data'
            The data directory

    OUTPUT
        A quadruple (X_train, X_test, Y_train, Y_test) consisting of either
        dense numpy arrays or sparse matrices, depending on whether fasttext
        vectors have been used or not
    '''
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from utils import get_cats, get_path

    # Load data
    df = pd.read_csv(get_path(data_dir) / f'{fname}.tsv', sep = '\t')
    cats = get_cats(data_dir = data_dir)['id']
    X, Y = df['text'], df.loc[:, cats].values
    del df

    # Create train- and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
        train_size = split_ratio, random_state = random_state)
    del X, Y

    if use_fasttext:
        from fasttext import load_model
        import numpy as np
        ft = load_model(str(get_path(data_dir) / 'fasttext_model.bin'))
        X_train = np.stack([ft.get_sentence_vector(doc.strip()) 
            for doc in X_train])
        X_test = np.stack([ft.get_sentence_vector(doc.strip()) 
            for doc in X_test])

    else:
        from sklearn.feature_extraction.text import CountVectorizer
        train_vocab = {tok for doc in X_train for tok in doc.split()}
        vectoriser = CountVectorizer().fit(train_vocab)
        X_train = vectoriser.transform(X_train)
        X_test = vectoriser.transform(X_test)

    return X_train, X_test, Y_train, Y_test

def evaluate(model, X_test, Y_test):
    ''' Compute the sample-average F1 score of the model on the test set. '''
    from sklearn.metrics import f1_score
    Y_hat = model.predict(X_test)
    nrows = Y_test.shape[0]
    score = sum(f1_score(Y_test[i, :], Y_hat[i, :]) for i in range(nrows))
    return round(score / nrows, 4)

def train_naive_bayes(X_train, X_test, Y_train, Y_test, workers: int = -1):
    ''' Train a multilabel naive bayes classifier. If the frequency vectors
    are used then the prior will be multinomial, and otherwise Gaussian. '''
    from sklearn.naive_bayes import MultinomialNB, GaussianNB
    from sklearn.multioutput import MultiOutputClassifier
    try:
        # If frequency vectors are used
        model = MultiOutputClassifier(MultinomialNB(), n_jobs = workers)
        model = model.fit(X_train, Y_train)
    except ValueError:
        # If fasttext vectors are used
        model = MultiOutputClassifier(GaussianNB(), n_jobs = workers)
        model = model.fit(X_train, Y_train)
    return model, evaluate(model, X_test, Y_test)

def train_svm(X_train, X_test, Y_train, Y_test, workers: int = -1):
    ''' Train a multilabel linear support vector machine. '''
    from sklearn.svm import LinearSVC
    from sklearn.multioutput import MultiOutputClassifier
    binary_model = LinearSVC()
    model = MultiOutputClassifier(binary_model, n_jobs = workers)
    model = model.fit(X_train, Y_train)
    return model, evaluate(model, X_test, Y_test)

def train_log_reg(X_train, X_test, Y_train, Y_test, workers: int = -1,
    max_iter: int = 5000):
    ''' Train a multilabel logistic regression classifier. '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    binary_model = LogisticRegression(max_iter = max_iter)
    model = MultiOutputClassifier(binary_model, n_jobs = workers)
    model = model.fit(X_train, Y_train)
    return model, evaluate(model, X_test, Y_test)

if __name__ == '__main__':

    print('### WITHOUT FASTTEXT VECTORS ###')
    data = load_data(use_fasttext = False)
    print('Naive Bayes score:', train_naive_bayes(*data)[1])
    print('Support vector machine score:', train_svm(*data)[1])
    print('Logistic regression score:', train_log_reg(*data)[1])

    print('### WITH FASTTEXT VECTORS ###')
    data = load_data(use_fasttext = True)
    print('Naive Bayes score:', train_naive_bayes(*data)[1])
    print('Support vector machine score:', train_svm(*data)[1])
    print('Logistic regression score:', train_log_reg(*data)[1])
