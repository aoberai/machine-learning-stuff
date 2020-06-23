import os
import tensorflow as tf

mnist = tf.keras.datasets.mnist
mnist_set = mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist_set
x_train, x_test = x_train / 255.0, x_test / 255.0


def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


model_path = "./digit_identification_model.h5"

model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
if model_path not in os.listdir():
    model.fit(x=x_train,
              y=y_train,
              epochs=5,
              validation_data=(x_test, y_test))
    model.save(model_path)

else:
    print("Model already made!")
    model = tf.keras.models.load_model(model_path)

print("-------------------------------------------------------------------")
print("\n")
print(model.summary())
print("\n")
loss, acc = model.evaluate(x=x_test,  y=y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
