import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import sys
from pathlib import Path
from datetime import datetime
from functools import reduce # used to calculate accuracy

# homegrown neural network
from NN import NeuralNetwork

# keras neural network packages
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


# set list of file_names
if len(sys.argv) > 1:
    file_names = sys.argv[1:]
else:
    file_names = [f'arxiv_sample_{i}' for i in
        [1000, 5000, 10000, 25000, 50000, 100000]]

home_dir = str(Path.home())
data_path = os.path.join(home_dir, "pCloudDrive", "Public Folder",
    "scholarly_data")

for file_name in file_names:
    print("------------------------------------")
    print(f"NOW PROCESSING: {file_name}")
    print("------------------------------------")

    full_path = os.path.join(data_path,
        f"{file_name}_model.pickle")
    if os.path.isfile(full_path):
        with open(full_path, 'rb') as pickle_in:
            nn_model = pickle.load(pickle_in)
        print("Model already trained.")
    else:
        # load test set
        full_path = os.path.join(data_path, f"arxiv_test_set.csv")
        df_1hot_agg = pd.read_csv(full_path)
        X_test = np.asarray(df_1hot_agg.iloc[:, :1024])
        Y_test = np.asarray(df_1hot_agg.iloc[:, 1024:])

        # fetch data
        full_path = os.path.join(data_path, f"{file_name}_1hot_agg.csv")
        df_1hot_agg = pd.read_csv(full_path)
        X = np.asarray(df_1hot_agg.iloc[:, :1024])
        Y = np.asarray(df_1hot_agg.iloc[:, 1024:])

        epochs = 4000

        nn_model = Sequential()
        nn_model.add(Dense(30, activation = 'tanh'))
        nn_model.add(Dense(5, activation = 'sigmoid'))
        nn_model.compile(
            loss = 'binary_crossentropy', 
            optimizer = SGD(lr = 0.1),
            metrics = ['accuracy']
            )

        H = nn_model.fit(X, Y, validation_data = (X_test, Y_test),
            epochs = epochs, batch_size = 64)

        # evaluate the network
        print("[INFO] evaluating network...")
        predictions = nn_model.predict(X_test, batch_size=32)
        print(classification_report(
                Y_test.argmax(axis=1),
                predictions.argmax(axis=1),
                target_names = ['physics', 'other', 'cs', 'maths', 'stats']
                ))

        # plot the training loss and accuracy
        N = np.arange(0, epochs)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H.history["loss"], label = "train_loss")
        plt.plot(N, H.history["val_loss"], label = "val_loss")
        plt.plot(N, H.history["acc"], label = "train_acc")
        plt.plot(N, H.history["val_acc"], label = "val_acc")
        plt.title("Training Loss and Accuracy (Simple NN)")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        full_path = os.path.join(data_path, f'{file_name}_plot.png')
        plt.savefig(full_path)
       
 
        ### HOMEGROWN NEURAL NETWORK ###
        
        #nn_model = NeuralNetwork(
        #    layer_dims = [30, 5],
        #    activations = ['tanh', 'sigmoid'],
        #    target_accuracy = 0.65,
        #    test_set = (X_test, Y_test),
        #    plots = True 
        #    )
        
        # time training for good measure
        #start_time = datetime.now()
    
        #nn_model.fit(X)
        #nn_model.train(X, Y)

        # calculate training accuracy
        #Yhat = np.squeeze(np.around(nn_model.predict(X), decimals = 0))
        #Yhat = Yhat.astype('int')
        #correct_predictions = np.sum(np.asarray(
        #    [reduce(lambda z, w: z and w, x) for x in np.equal(Y.T, Yhat.T)]))
        #train_accuracy = correct_predictions / X.shape[1]

        ## calculate test accuracy
        #Yhat = np.squeeze(np.around(nn_model.predict(X_test), decimals = 0))
        #Yhat = Yhat.astype('int')
        #correct_predictions = np.sum(np.asarray([reduce(
        #    lambda z, w: z and w, x) for x in np.equal(Y_test.T, Yhat.T)]))
        #test_accuracy = correct_predictions / X_test.shape[1]

        #print("Training complete!")
        #print(f"Training accuracy: {np.around(train_accuracy * 100, 2)}%")
        #print(f"Test accuracy: {np.around(test_accuracy * 100, 2)}%")
        #print(f"Time spent: {datetime.now() - start_time}")


        # save model
        full_path = os.path.join(data_path, f'{file_name}_model.pickle')
        with open(full_path, 'wb') as pickle_out:
            pickle.dump(nn_model, pickle_out)
        
