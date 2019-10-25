from keras.models import Sequential
from keras.layers import Dense


nb_hidden_nodes = 2

model = Sequential()
model.add(Dense(nb_hidden_nodes, input_shape=(5, ), use_bias=True, activation='sigmoid'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=5, batch_size=32)