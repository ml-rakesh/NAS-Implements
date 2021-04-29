import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import autokeras as ak
import datasets
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

x_train, x_test, y_train, y_test = datasets.load_data()


input_node = ak.StructuredDataInput()
# x1 = ak.StructuredDataBlock(categorical_encoding=True, normalize=False)(input_node)
x1 = ak.Normalization()(input_node)
x1 = ak.DenseBlock(num_layers=1, num_units=32)(x1)
x1 = ak.DenseBlock(num_layers=1, num_units=32, dropout=0.1, use_batchnorm=False)(x1)
output_node = ak.ClassificationHead(loss = "categorical_crossentropy", num_classes=2)(x1)

import kerastuner
auto_model = ak.AutoModel(inputs=input_node,
                          outputs=output_node,
#                           num_classes=2,
                          overwrite=True,
                          max_trials=3,
                          objective=kerastuner.Objective('val_auroc', direction='max'),
                          metrics = [tf.keras.metrics.AUC(name="auroc", curve="ROC")])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
auto_model.fit(x_train,
               y_train,
               epochs = 20,
               callbacks = [es],
               batch_size = 32,
               class_weight = {0: 1.0, 1: 2.3})

model = auto_model.export_model()
model.summary()
y_pred_temp = model.predict(x_test, batch_size=32)
y_pred_temp = y_pred_temp.reshape(1, -1)[0]
y_pred = [0 if x < 0.5 else 1 for x in y_pred_temp]

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred))