
import numpy as np
import pandas as pd
import tensorflow as tf

import datasets
import kerastuner
import autokeras as ak
from sklearn.metrics import classification_report, confusion_matrix

x_train, x_test, y_train, y_test = datasets.load_data()

# Initialize the structured data classifier.
clf = ak.StructuredDataClassifier(
    num_classes=2,
    overwrite=True,
    max_trials=4,
    loss = "categorical_crossentropy",
    objective=kerastuner.Objective('val_auroc', direction='max'),
    metrics = [tf.keras.metrics.AUC(name="auroc", curve="ROC")]
)

clf.fit(x = x_train,
    y = y_train,
    validation_data = (x_test, y_test),
    epochs=15,
    class_weight={0:0.5,1:2.5},
    verbose = 2
)

predicted_y = clf.predict(x_test)
actual_y = y_test

print(classification_report(actual_y, predicted_y[:, 0]))