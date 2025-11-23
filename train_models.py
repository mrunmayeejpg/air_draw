import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

print("Loading MNIST...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1,28,28,1) / 255.
x_test  = x_test.reshape(-1,28,28,1) / 255.

for i in range(3):
    print(f"Training model {i+1}/3...")
    model = create_model()
    model.fit(x_train, y_train, epochs=2, batch_size=64, validation_split=0.1)
    model.save(f"models/mnist_model_{i}.h5")
    print(f"Saved: models/mnist_model_{i}.h5")

print("All models trained & saved!")
