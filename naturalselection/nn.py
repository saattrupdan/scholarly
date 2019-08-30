import numpy as np
import os
from functools import partial 
from naturalselection.core import Genus, Population

class FNN(Genus):
    ''' Feedforward fully connected neural network genus.

    INPUT:
        (int) max_number_of_hidden_layers
        (bool) uniform_layers: whether all hidden layers should
               have the same amount of neurons and dropout
        (iterable) dropout: values for dropout
        (iterable) neurons_per_hidden_layer = neurons in hidden layers
        (iterable) optimizer: keras optimizers
        (iterable) hidden_activation: keras activation functions
        (iterable) batch_size: batch sizes
        (iterable) initializer: keras initializers
        '''
    def __init__(self,
        max_number_of_hidden_layers = 5,
        uniform_layers = False,
        dropout = np.arange(0, 0.6, 0.1),
        neurons = np.array([2 ** n for n in range(4, 13)]),
        optimizer = np.array(['sgd', 'rmsprop', 'adagrad', 'adadelta',
                              'adamax', 'adam', 'nadam']),
        hidden_activation = np.array(['relu', 'elu', 'softplus', 'softsign']),
        batch_size = np.array([2 ** n for n in range(4, 12)]),
        initializer = np.array(['lecun_uniform', 'lecun_normal',
                                'glorot_uniform', 'glorot_normal',
                                'he_uniform', 'he_normal'])):

        self.optimizer = np.unique(np.asarray(optimizer))
        self.hidden_activation = np.unique(np.asarray(hidden_activation))
        self.batch_size = np.unique(np.asarray(batch_size))
        self.initializer = np.unique(np.asarray(initializer))
        self.input_dropout = np.unique(np.asarray(dropout))

        if uniform_layers:
            self.neurons = neurons
            self.dropout = dropout
            self.number_of_hidden_layers = \
                np.arange(1, max_number_of_hidden_layers + 1)
        else:
            neurons = np.unique(np.append(neurons, 0))
            dropout = np.around(np.unique(np.append(dropout, 0)), 2)
            layer_info = {}
            for layer_idx in range(max_number_of_hidden_layers):
                layer_info["neurons{}".format(layer_idx)] = neurons
                layer_info["dropout{}".format(layer_idx)] = dropout
            self.__dict__.update(layer_info)

class FNNs(Population):
    def __init__(self, 
        train_val_sets,
        size = 50, 
        initial_genome = None,
        loss_fn = 'binary_crossentropy',
        number_of_features = 'infer', 
        number_of_labels = 'infer',
        score = 'accuracy', 
        output_activation = 'sigmoid',
        max_epochs = 1000000, 
        patience = 5, 
        min_change = 1e-4,
        max_training_time = None, 
        max_number_of_hidden_layers = 5,
        uniform_layers = False,
        dropout = np.arange(0, 0.6, 0.1),
        neurons = np.array([2 ** n for n in range(4, 13)]),
        optimizer = np.array(['sgd', 'rmsprop', 'adagrad', 'adadelta',
                              'adamax', 'adam', 'nadam']),
        hidden_activation = np.array(['relu', 'elu', 'softplus',
                                      'softsign']),
        batch_size = np.array([2 ** n for n in range(4, 12)]),
        initializer = np.array(['lecun_uniform', 'lecun_normal',
                                'glorot_uniform', 'glorot_normal',
                                'he_uniform', 'he_normal']),
        verbose = 0):

        self.train_val_sets                 = train_val_sets
        self.size                           = size
        self.initial_genome                 = initial_genome
        self.loss_fn                        = loss_fn
        self.number_of_features             = number_of_features
        self.number_of_labels               = number_of_labels
        self.score                          = score
        self.output_activation              = output_activation
        self.max_epochs                     = max_epochs 
        self.patience                       = patience
        self.min_change                     = min_change
        self.max_training_time              = max_training_time
        self.max_number_of_hidden_layers    = max_number_of_hidden_layers
        self.uniform_layers                 = uniform_layers
        self.dropout                        = np.asarray(dropout)
        self.neurons                        = np.asarray(neurons)
        self.optimizer                      = np.asarray(optimizer)
        self.hidden_activation              = np.asarray(hidden_activation)
        self.batch_size                     = np.asarray(batch_size)
        self.initializer                    = np.asarray(initializer)
        self.verbose                        = verbose

        self.genus = FNN(
            max_number_of_hidden_layers = self.max_number_of_hidden_layers,
            uniform_layers              = self.uniform_layers,
            dropout                     = self.dropout,
            neurons                     = self.neurons,
            optimizer                   = self.optimizer,
            hidden_activation           = self.hidden_activation,
            batch_size                  = self.batch_size,
            initializer                 = self.initializer
            )

        self.fitness_fn = partial(
            self.train_fnn,
            max_epochs          = self.max_epochs,
            patience            = self.patience,
            min_change          = self.min_change,
            max_training_time   = self.max_training_time,
            verbose             = self.verbose,
            file_name           = None
            )

        if self.initial_genome:
            self.population = np.array(
                [Organism(self.genus, **self.initial_genome)
                for _ in range(self.size)]
                )
        else:
            self.population = self.genus.create_organisms(self.size)

        self.fittest = np.random.choice(self.population)

    def train_best(self, max_epochs = 1000000, min_change = 1e-4,
        patience = 5, max_training_time = None, file_name = None):

        best_fnn = self.fittest        
        score = self.train_fnn(
            fnn                 = best_fnn,
            max_epochs          = max_epochs,
            patience            = patience,
            min_change          = min_change,
            max_training_time   = max_training_time,
            verbose             = 1,
            file_name           = file_name
            )
        return score

    def train_fnn(self, fnn, max_epochs = 1000000, patience = 5,
        min_change = 1e-4, max_training_time = None, verbose = False,
        file_name = None):
        ''' Train a feedforward neural network and output the score.
        
        INPUT
            (FNN) fnn: a feedforward neural network genus
            (int) max_epochs: maximum number of epochs to train for
            (int) patience: number of epochs with no progress above min_change
            (float) min_change: everything below this number won't count as a
                    change in the score
            (int) max_training_time: maximum number of seconds to train for,
                  also training the final epoch after the time has run out
            (int) verbose: verbosity mode

        OUTPUT
            (float) the score of the neural network
        '''

        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras.callbacks import Callback, EarlyStopping
        from tensorflow.keras import backend as K
        from tensorflow.python.util import deprecation
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import f1_score, precision_score, recall_score

        # Used when building network
        from itertools import count

        # Used for the EarlierStopping callback
        import time

        # Used for the TQDMCallback callback
        from tqdm import tqdm

        # Used for constructing generator
        from itertools import tee

        # Suppress deprecation warnings
        deprecation._PRINT_DEPRECATION_WARNINGS = False

        # Suppress tensorflow warnings and infos
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        class TQDMCallback(Callback):
            '''
            Callback to enable tqdm integration.
            Source: https://github.com/bstriner/keras-tqdm
            '''

            def __init__(self, outer_description = "Training",
                inner_description_initial = "Epoch: {epoch}",
                inner_description_update = "Epoch: {epoch} - {metrics}",
                metric_format = "{name}: {value:0.3f}",
                separator = ", ",
                leave_inner = True,
                leave_outer = True,
                show_inner = True,
                show_outer = True,
                output_file = None,
                initial = 0):

                self.outer_description          = outer_description
                self.inner_description_initial  = inner_description_initial
                self.inner_description_update   = inner_description_update
                self.metric_format              = metric_format
                self.separator                  = separator
                self.leave_inner                = leave_inner
                self.leave_outer                = leave_outer
                self.show_inner                 = show_inner
                self.show_outer                 = show_outer
                self.output_file                = output_file
                self.tqdm_outer                 = None
                self.tqdm_inner                 = None
                self.epoch                      = None
                self.running_logs               = None
                self.inner_count                = None
                self.initial                    = initial

            def build_tqdm(self, desc, total, leave, initial = 0):
                """
                Extension point. Override to provide custom options to tqdm
                initializer.
                """
                return tqdm(desc = desc, total = total, leave = leave, 
                    file = self.output_file, initial = initial)

            def build_tqdm_outer(self, desc, total):
                """
                Extension point. Override to provide custom options to outer
                progress bars (Epoch loop)
                """
                return self.build_tqdm(desc = desc, total = total,
                    leave = self.leave_outer, initial = self.initial)

            def build_tqdm_inner(self, desc, total):
                """
                Extension point. Override to provide custom options to inner
                progress bars (Batch loop)
                """
                return self.build_tqdm(desc = desc, total = total,
                    leave = self.leave_inner)

            def on_epoch_begin(self, epoch, logs = {}):
                self.epoch = epoch
                desc = self.inner_description_initial.format(
                    epoch = self.epoch)
                self.mode = 0  # samples
                if 'samples' in self.params:
                    self.inner_total = self.params['samples']
                elif 'nb_sample' in self.params:
                    self.inner_total = self.params['nb_sample']
                else:
                    self.mode = 1  # steps
                    self.inner_total = self.params['steps']
                if self.show_inner:
                    self.tqdm_inner = self.build_tqdm_inner(desc = desc,
                        total = self.inner_total)
                self.inner_count = 0
                self.running_logs = {}

            def on_epoch_end(self, epoch, logs = {}):
                metrics = self.format_metrics(logs)
                desc = self.inner_description_update.format(epoch = epoch,
                    metrics = metrics)
                if self.show_inner:
                    self.tqdm_inner.desc = desc
                    # set miniters and mininterval to 0 so last update shows 
                    self.tqdm_inner.miniters = 0
                    self.tqdm_inner.mininterval = 0
                    self.tqdm_inner.update(self.inner_total-self.tqdm_inner.n)
                    self.tqdm_inner.close()
                if self.show_outer:
                    self.tqdm_outer.update(1)

            def on_batch_begin(self, batch, logs = {}):
                pass

            def on_batch_end(self, batch, logs = {}):
                if self.mode == 0:
                    update = logs['size']
                else:
                    update = 1
                self.inner_count += update
                if self.inner_count < self.inner_total:
                    self.append_logs(logs)
                    metrics = self.format_metrics(self.running_logs)
                    desc = self.inner_description_update.format(
                        epoch = self.epoch, metrics = metrics)
                    if self.show_inner:
                        self.tqdm_inner.desc = desc
                        self.tqdm_inner.update(update)

            def on_train_begin(self, logs = {}):
                if self.show_outer:
                    epochs = (self.params['epochs'] if 'epochs' in self.params
                              else self.params['nb_epoch'])
                    self.tqdm_outer = self.build_tqdm_outer(
                        desc = self.outer_description, total = epochs)

            def on_train_end(self, logs = {}):
                if self.show_outer:
                    self.tqdm_outer.close()

            def append_logs(self, logs):
                metrics = self.params['metrics']
                for metric, value in logs.items():
                    if metric in metrics:
                        if metric in self.running_logs:
                            self.running_logs[metric].append(value[()])
                        else:
                            self.running_logs[metric] = [value[()]]

            def format_metrics(self, logs):
                metrics = self.params['metrics']
                strings = [self.metric_format.format(name = metric,
                    value = np.mean(logs[metric], axis = None))
                    for metric in metrics if metric in logs]
                return self.separator.join(strings)

        class EarlierStopping(EarlyStopping):
            '''
            Callback to stop training when enough time has passed.
            Source: https://github.com/keras-team/keras-contrib/issues/87

            INPUT
                (int) seconds: maximum time before stopping.
                (int) verbose: verbosity mode.
            '''
            def __init__(self, seconds = None, **kwargs):
                super().__init__(**kwargs)
                self.start_time = 0
                self.seconds = seconds

            def on_train_begin(self, logs = {}):
                self.start_time = time.time()
                super().on_train_begin(logs)

            def on_batch_end(self, batch, logs = {}):
                if self.seconds and time.time()-self.start_time > self.seconds:
                    self.model.stop_training = True
                    if self.verbose:
                        print('Stopping after {} seconds.'\
                            .format(self.seconds))

            def on_epoch_end(self, epoch, logs = {}):
                if self.seconds and time.time()-self.start_time > self.seconds:
                    self.model.stop_training = True
                    if self.restore_best_weights and self.best_weights:
                        self.model.set_weights(self.best_weights) 
                    if self.verbose:
                        print('Stopping after {} seconds.'.\
                            format(self.seconds))

                # This restores best weights if we're beyond the first epoch
                if logs.get(self.monitor):
                    super().on_epoch_end(epoch, logs)

        X_train, Y_train, X_val, Y_val = self.train_val_sets

        if self.number_of_features == 'infer':
            self.number_of_features = X_train.shape[1]
        if self.number_of_labels == 'infer':
            self.number_of_labels = Y_train.shape[1]

        inputs = Input(shape = (self.number_of_features,))
        x = Dropout(fnn.input_dropout)(inputs)

        if self.uniform_layers:
            for _ in range(fnn.number_of_hidden_layers):
                x = Dense(fnn.neurons, activation = fnn.hidden_activation,
                    kernel_initializer = fnn.initializer)(x)
                x = Dropout(fnn.dropout)(x)
        else:
            for i in count():
                try:
                    neurons = fnn.__dict__["neurons{}".format(i)]
                    if neurons:
                        x = Dense(neurons, activation = fnn.hidden_activation,
                            kernel_initializer = fnn.initializer)(x)
                    dropout = fnn.__dict__["dropout{}".format(i)]
                    if dropout:
                        x = Dropout(dropout)(x)
                except:
                    break
        outputs = Dense(self.number_of_labels,
            activation = self.output_activation,
            kernel_initializer = fnn.initializer)(x)
        nn = Model(inputs = inputs, outputs = outputs)

        nn.compile(
            loss = self.loss_fn,
            optimizer = fnn.optimizer,
            )

        early_stopping = EarlierStopping(
            monitor = 'val_loss',
            patience = patience,
            min_delta = min_change,
            restore_best_weights = True,
            seconds = max_training_time
            )

        tqdm_callback = TQDMCallback(
            show_outer = False
        )

        callbacks = [early_stopping]
        if verbose:
            callbacks.append(tqdm_callback)

        H = nn.fit(
            X_train,
            Y_train,
            batch_size = fnn.batch_size,
            validation_data = (X_val, Y_val),
            epochs = max_epochs,
            callbacks = callbacks,
            verbose = 0
            )

        if file_name:
            nn.save("{}.h5".format(file_name))

        if Y_val.shape[1] > 1:
            average = 'micro'
        else:
            average = None

        Y_hat = nn.predict(X_val, batch_size = 32)
        if self.score == 'accuracy':
            Y_hat = np.greater(Y_hat, 0.5)
            fitness = accuracy_score(Y_val, Y_hat)
        elif self.score == 'f1':
            Y_hat = np.greater(Y_hat, 0.5)
            fitness = f1_score(Y_val, Y_hat, average = average)
        elif self.score == 'precision':
            Y_hat = np.greater(Y_hat, 0.5)
            fitness = precision_score(Y_val, Y_hat, average = average)
        elif self.score == 'recall':
            Y_hat = np.greater(Y_hat, 0.5)
            fitness = recall_score(Y_val, Y_hat, average = average)
        elif self.score == 'loss':
            fitness = np.divide(1, nn.evaluate(X_val, Y_val))
        else:
            # Custom scoring function
            fitness = self.score(Y_val, Y_hat)
        
        # Clear tensorflow session to avoid memory leak
        K.clear_session()
            
        return fitness


def __main__():
    pass
