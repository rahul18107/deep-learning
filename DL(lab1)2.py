import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',
optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])
model.fit(X, y, epochs=1000, verbose=0) 

_, acc = model.evaluate(X, y, verbose=0)
print(f"Accuracy: {acc*100:.2f}%")
print(np.round(model.predict(X)))