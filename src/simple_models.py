from utils import get_cats, get_path, boolean

def load_data(fname: str = 'arxiv_data_pp', data_dir: str = '.data',
    split_ratio: float = 0.98, random_state: int = 42, 
    use_fasttext: bool = False, use_ngrams: bool = False, **kwargs):
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
        use_ngrams: bool = False
            Whether to use word ngrams in the word vectors. Increases loading
            time.
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

    # Load data
    df = pd.read_csv(get_path(data_dir) / f'{fname}.tsv', sep = '\t')
    cats = get_cats(data_dir = data_dir)['id']
    X, Y = df['text'], df.loc[:, cats].values
    del df

    # Convert into ngrams
    if use_ngrams:
        from gensim.models.phrases import Phrases, Phraser
        phrases = Phrases((x.split() for x in X))
        phraser = Phraser(phrases)
        X = X.apply(lambda x: ' '.join(phraser[x.split()]))
        del phrases, phraser

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
    ''' Compute the score of a model.
    
    INPUT
        model
            A scikit-learn machine learning model
        X_test
            The feature vectors for inference, of shape (ntest, vector_dim)
        Y_test: numpy.ndarray
            The label vectors for inference, of shape (ntest, ncats)

    OUTPUT
        The sample-average F1 score of the model on the test set
    '''
    from sklearn.metrics import f1_score
    Y_hat = model.predict(X_test)
    nrows = Y_test.shape[0]
    score = sum(f1_score(Y_test[i, :], Y_hat[i, :]) for i in range(nrows))
    return round(score / nrows, 4)

def train_model(X_train, X_test, Y_train, Y_test, workers: int = -1,
    model_type: str = 'naive_bayes', max_iter_factor: int = 10, 
    random_state: int = 42,  **kwargs):
    ''' Trains a simple machine learning model on the dataset. 
    
    INPUT
        X_train: numpy.ndarray
            The feature vectors for training, of shape (ntrain, vector_dim)
        X_test: numpy.ndarray
            The feature vectors for inference, of shape (ntest, vector_dim)
        Y_train: numpy.ndarray
            The label vectors for training, of shape (ntrain, ncats)
        Y_test: numpy.ndarray
            The label vectors for inference, of shape (ntest, ncats)
        workers: int = -1
            The number of processes to run in parallel. Defaults to
            running a process for each CPU core
        model_type: str = 'naive_bayes'
            What model to train. Can be chosen among 'naive_bayes', 'svm'
            and 'logreg'
        max_iter_factor: int = 10:
            A factor determining the number of iterations to train for, when
            training the SVM or logistic regression classifier. It will
            multiply this factor by the default number of iterations for the
            classifier, which is 1000 for SVM and 100 for logistic regression
        random_state: int = 42
            Random state for reproducibility

    OUTPUT
        A pair (model, score), with model being the trained model and score
        the sample-average F1 score on the test set
    '''
    
    if model_type == 'naive_bayes':
        from sklearn.naive_bayes import MultinomialNB, GaussianNB
        from sklearn.multioutput import MultiOutputClassifier

        # If frequency vectors are used
        if X_train.shape[1] > 100:
            model = MultiOutputClassifier(MultinomialNB(), n_jobs = workers)

        # If fasttext vectors are used
        else:
            model = MultiOutputClassifier(GaussianNB(), n_jobs = workers)

    elif model_type == 'svm':
        from sklearn.svm import LinearSVC
        from sklearn.multioutput import MultiOutputClassifier
        binary_model = LinearSVC(
            max_iter = 1000 * max_iter_factor, 
            random_state = random_state,
            verbose = 1
        )
        model = MultiOutputClassifier(binary_model, n_jobs = workers)

    elif model_type == 'logreg':
        from sklearn.linear_model import LogisticRegression
        from sklearn.multioutput import MultiOutputClassifier
        binary_model = LogisticRegression(
            max_iter = 100 * max_iter_factor, 
            random_state = random_state,
            verbose = 1
        )
        model = MultiOutputClassifier(binary_model, n_jobs = workers)

    model = model.fit(X_train, Y_train)
    return model, evaluate(model, X_test, Y_test)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from multiprocessing import cpu_count

    parser = ArgumentParser()
    parser.add_argument('--model_type', default = 'naive_bayes')
    parser.add_argument('--fname', default = 'arxiv_data_pp')
    parser.add_argument('--use_fasttext', type = boolean, default = False)
    parser.add_argument('--split_ratio', type = float, default = 0.98)
    parser.add_argument('--workers', type = int, default = -1)
    parser.add_argument('--max_iter_factor', type = int, default = 10)
    parser.add_argument('--use_ngrams', type = boolean, default = False)
    parser.add_argument('--data_dir', default = '.data')
    args = vars(parser.parse_args())

    workers = cpu_count() if args['workers'] == -1 else args['workers']
    vectors = 'FastText' if args['use_fasttext'] else 'frequency'
    worker_tense = 'workers' if workers > 1 else 'worker'
    print(f'Training {args["model_type"]} on {args["fname"]} with '\
          f'{vectors} vectors and {workers} {worker_tense}.')

    data = load_data(**args)
    print(train_model(*data, **args)[1])
