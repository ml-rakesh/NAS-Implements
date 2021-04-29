from deephyper.problem import NaProblem
from nas_1.polynome2.load_data import load_data
from nas_1.polynome2.search_space import create_search_space
from deephyper.nas.preprocessing import minmaxstdscaler,stdscaler
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)

# from deephyper.nas.train_utils import *
from sklearn import metrics

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

Problem.preprocessing(stdscaler)

Problem.search_space(create_search_space, num_layers=6)

Problem.hyperparameters(
    batch_size=128,
    learning_rate=0.001,
    optimizer='adam',
    num_epochs=20,
    callbacks=dict(
        EarlyStopping=dict(
            monitor='val_loss', # or 'val_r2' or 'val_acc' ?
            # mode='max',
            verbose=0,
            patience=5,
            restore_best_weights=True
        )
    )
)

Problem.loss('categorical_crossentropy') # or 'mse' ?

def AUC(y_true,y_pred):
    y_true = tf.argmax(y_true, axis = 1)
    m = tf.keras.metrics.AUC(num_thresholds=3)
    m.update_state(y_true, y_pred[:,1])
    return m.result()

# for multi class multi label
# def f1_score(y_true,y_pred):

#     def recall(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall

#     def precision(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision
#     actual = y_true
#     predicted = 1 if y_pred>=0.5 else 0
#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))


Problem.metrics([AUC]) # or 'acc' ?

Problem.objective('val_AUC__max') # or 'val_r2__last' ?

# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == '__main__':
    print(Problem)