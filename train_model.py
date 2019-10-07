import numpy as np
import os

# NaturalSelection has found the following, giving val F1-score 88.9%:
#   activation  = relu
#   optimizer   = adam
#   initializer = he_normal
#   neurons     = [512, 512, 64, 1024, 128]
#   dropouts    = [10%, 0%, 10%, 20%, 40%, 0%]
#   pos_weight  = ?? (used 1)
#   batch size  = 1024 (scoring after ten mins)
#
#x = Dropout(0.1)(inputs)
#x = Dense(512, activation = self.activation,
#    kernel_initializer = self.initializer)(x)
#x = Dense(512, activation = self.activation,
#    kernel_initializer = self.initializer)(x)
#x = Dropout(0.1)(x)
#x = Dense(64, activation = self.activation,
#    kernel_initializer = self.initializer)(x)
#x = Dropout(0.2)(x)
#x = Dense(1024, activation = self.activation,
#    kernel_initializer = self.initializer)(x)
#x = Dropout(0.4)(x)
#x = Dense(128, activation = self.activation,
#    kernel_initializer = self.initializer)(x)


class Model():
    
    def __init__(self, activation = 'relu', 
        learning_rate = 0.001, fst_moment = 0.9, snd_moment = 0.999,
        decay = 0, nesterov = True, loss = 'binary_crossentropy'):

        from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Nadam
        from tensorflow.keras.initializers import VarianceScaling

        self.activation = activation
        self.loss = loss
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

        if snd_moment:
            if fst_moment:
                if nesterov:
                    self.optimizer = Nadam(lr = learning_rate, 
                        beta_1 = fst_moment, beta_2 = snd_moment,
                        schedule_decay = decay)
                else:
                    self.optimizer = Adam(lr = learning_rate, 
                        beta_1 = fst_moment, beta_2 = snd_moment,
                        decay = decay)
            else:
                self.optimizer = RMSprop(lr = learning_rate, 
                    rho = snd_moment, decay = decay)
        else:
            self.optimizer = SGD(lr = learning_rate,
                momentum = fst_moment, nesterov = nesterov,
                decay = decay)

        if self.activation == 'relu':
            # This initialisation is specific to ReLU to ensure that the
            # variances of the weights and gradients neither vanishes
            # nor explodes.
            # Source: He et al, "Delving Deep into Rectifiers: Surpassing
            # Human-Level Performance on ImageNet Classification"
            self.initializer = 'he_normal'
        elif self.activation == 'elu':
            # This initialisation is specific to ELU, where the the 1.55
            # value is derived as in the He et al paper above, but further-
            # more assumes that the input values follow a standard
            # normal distribution, so this is only an approximation.
            # Source: https://stats.stackexchange.com/a/320443/255420
            self.initializer = VarianceScaling(scale = 1.55,
                distribution = 'normal', mode = 'fan_in')
        else:
            # This initialisation is good for activations that are symmetric
            # around zero, like sigmoid, softmax and tanh.
            # Source: Glorot and Bengio, "Understanding the difficulty of
            # training deep feedforward neural networks"
            self.initializer = 'glorot_normal'

    def compile(self, file_name, data_path):
        ''' Load text data and labels. '''

        from tensorflow.keras.models import Model, load_model
        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras.layers import Dropout, BatchNormalization
        import pandas as pd

        vec_path = os.path.join(data_path, f'{file_name}_agg_vec.npy')
        self.x_train = np.load(vec_path)

        labels_path = os.path.join(data_path, f'{file_name}_agg.csv')
        df = pd.read_csv(labels_path, usecols = range(1, 7), 
            squeeze = True, header = None)
        self.y_train = np.array(df)
    
        self.sample_weight = np.log(1 + df.dot(sum(df.sum()) / df.sum()))

        vec_path = os.path.join(data_path, f'arxiv_val_agg_vec.npy')
        self.x_val = np.load(vec_path)

        labels_path = os.path.join(data_path, f'arxiv_val_agg.csv')
        self.y_val = np.asarray(pd.read_csv(labels_path, 
            usecols = range(1, 7), squeeze = True, header = None))

        input_shape = self.x_train.shape[1:]
        nm_labels = self.y_train.shape[1]

        inputs = Input(shape = input_shape)
        x = inputs
        x = BatchNormalization()(x)
        x = Dense(128, activation = self.activation,
            kernel_initializer = self.initializer)(x)
        x = Dense(nm_labels, activation = 'sigmoid',
            kernel_initializer = 'glorot_normal')(x)
        self.model = Model(inputs = inputs, outputs = x)
        self.model.compile(optimizer = self.optimizer, loss = self.loss, 
            metrics = ['accuracy'])

        return self

    def train(self, epochs = 10, early_stopping = True, min_delta = 0, 
        patience = 0, monitor = 'val_acc', plot = False, batch_size = 32):
        ''' Train the neural network. '''

        from tensorflow.keras.callbacks import EarlyStopping
        import matplotlib.pyplot as plt

        callbacks = []
        if early_stopping:
            es = EarlyStopping(monitor = monitor, patience = patience, 
                restore_best_weights = True, min_delta = min_delta)
            callbacks.append(es)

        history = self.model.fit(self.x_train, self.y_train, epochs = epochs,
            callbacks = callbacks, validation_data = (self.x_val, self.y_val),
            sample_weight = self.sample_weight, batch_size = batch_size)

        if plot:
            plt.plot(history.history['acc'], label='train')
            plt.plot(history.history['val_acc'], label='test')
            plt.legend()
            plt.show()

        return self

    def predict(self, x, threshold = 0.5):
        ''' Predict classes that x belongs to. '''
        y_pred = self.model.predict(x, batch_size = 32)
        threshold = np.max(y_pred) * threshold
        return np.greater(y_pred, threshold).astype(int)

    def predict_proba(self, x_val):
        ''' Predict probabilities of x belonging to classes. '''
        return self.model.predict(x_val)

    def score(self, threshold = 0.5):
        ''' Get micro-average F1-score. '''
        from sklearn.metrics import f1_score
        y_pred = self.predict(self.x_val, threshold = 0.5)
        return f1_score(self.y_val, y_pred, average = 'micro')

    def report(self, threshold = 0.5):
        from sklearn.metrics import classification_report
        y_pred = self.predict(self.x_val, threshold = 0.5)
        target_names = ['physics', 'compsci', 'maths', 
                        'quantbio', 'quantfin', 'stats']
        return classification_report(self.y_val, y_pred, 
            labels = range(0, 6), target_names = target_names)

    def save_nn(self, file_name):
        ''' Save model. '''
        self.model.save(file_name)
        return self

    def save(self, directory = None, file_name = 'nn_model'):
        import pickle
        self.model.save(file_name + '.h5')
        with open(file_name + '.pkl', 'wb') as file_out:
            info = {key: val for (key, val) in self.__dict__.items()
                    if key != 'model'}
            print(info)
            pickle.dump(info, file_out)
        return self

    @classmethod
    def load(cls, file_name = 'nn_model'):
        import pickle
        from tensorflow.keras.models import load_model
        with open(file_name + '.pkl', 'rb') as file_in:
            info = pickle.load(file_in)
        model = Model()
        model.__dict__.update(info)
        model.model = load_model(file_name + '.h5')
        return model


if __name__ == '__main__':
    
    import pathlib
    from tensorflow.keras.models import load_model
    data_path = os.path.join(str(pathlib.Path.home()), 'pCloudDrive', 
        'public_folder', 'scholarly_data')

    file_name = 'arxiv'

    #model = Model(file_name, data_path)
    #model.train(epochs = 100, patience = 5)
    #model.save()

    model = Model().compile(file_name, data_path)
    model.save()
    print(model.report())
