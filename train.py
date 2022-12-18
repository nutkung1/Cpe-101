import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
# from keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten, TimeDistributed, Dropout
from keras.callbacks import TensorBoard
from sklearn.svm import SVC
import matplotlib.pyplot as plt
label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(X_train.shape)
print(y_train.shape)
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = keras.Sequential([
        keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=(50, 63, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(256, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(256, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dense(256),
        keras.layers.Dense(128),
        keras.layers.Flatten(),
        keras.layers.Dense(actions.shape[0], activation='softmax')
    ])
# model = Sequential()
# model.add(LSTM(256, return_sequences=True, activation='relu', input_shape=(30,63)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))
res = [.7, 0.2, 0.1]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train ,batch_size=128, epochs=6,validation_data=(X_test, y_test))
model.summary()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
Y_pred = (model.predict(X_test) > 0.6).astype("int32")
y_pred = np.argmax(Y_pred, axis =1)
p=model.predict(X_test)
print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names = actions))
print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
model.save('model1.h5')
