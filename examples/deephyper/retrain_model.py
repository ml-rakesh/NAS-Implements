import numpy as np
import pandas as pd
import tensorflow as tf
from nas_problems.polynome2.load_data import load_data
from nas_problems.polynome2.problem import Problem
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("nas_problems/polynome2/results.csv")
i_best = df.objective.argmax()
row_best = df.iloc[i_best].tolist()
objective = row_best[-2]
arch_seq = row_best[:-2]
arch_seq = [int(i) for i in arch_seq]
model = Problem.get_keras_model(arch_seq)
model.save('best_model.h5')


(x1,y1),(x2,y2) = load_data()

# tf.config.run_functions_eagerly(True)
def AUC(y_true,y_pred):
    y_true = tf.argmax(y_true, axis = 1)
    m = tf.keras.metrics.AUC(num_thresholds=3)
    m.update_state(y_true, y_pred[:,1])
    return m.result()
adam = tf.keras.optimizers.Adamax(learning_rate=0.0001)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, min_lr = 0.0000001)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,restore_best_weights=True)

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=[AUC])
out = model.fit(x1,y1, validation_data=(x2,y2), epochs=50, batch_size=128, verbose=2,
                class_weight={0:0.5,1:1.5},callbacks=[reduce_lr])

pred_y = model.predict(x2)
pred_y_argmax = np.argmax(pred_y, axis = 1)

print(classification_report(np.argmax(y2, axis=1), pred_y_argmax))