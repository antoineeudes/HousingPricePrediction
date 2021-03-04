# %%
import os
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split


# %%
dataset = loadtxt(os.path.join("dataset", "dataset.csv"), delimiter=",", skiprows=1)
X = dataset[:, 1:]
y = dataset[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
model = Sequential()
model.add(Dense(50, input_dim=22, activation="relu"))
model.add(Dense(1, input_dim=50))

# %%
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
log_dir = os.path.join("logs","fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# %%
model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=100,
    validation_data=(X_test, y_test),
    callbacks=[tensorboard_callback],
)

# %%
loss, _ = model.evaluate(X_test, y_test)
print("Loss: %.2f" % loss)

if not os.path.exists('models'):
    os.makedirs('models')

model.save(os.path.join("models", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
