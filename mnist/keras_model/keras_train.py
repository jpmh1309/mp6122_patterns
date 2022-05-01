from keras import layers, models
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Preprocessing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create the model
model = models.Sequential()
model.add(layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(64, 3, activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Print summary of the model
model.summary()

# Train the model
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

batch_size = 128
epochs = 15

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Test the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save model and weights to separated files.
with open("model.json", "w") as file:
    file.write(model.to_json())
model.save_weights("weights.h5")

# Save model and weights to the same file.
model.save('model.h5', include_optimizer=False)