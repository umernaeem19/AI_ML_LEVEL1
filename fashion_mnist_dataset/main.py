import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
##print(tf.__version__)

fmnist = tf.keras.datasets.fashion_mnist

#split data into training and test sets
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

index = 0

#number of characters per row when printing
np.set_printoptions(linewidth=320)

#label and image
# print(f'LABEL: {training_labels[index]}')
# print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')

# Visualize the image
#plt.imshow(training_images[index])


# Normalizing the data
training_images  = training_images / 255.0
test_images = test_images / 255.0

accuracy = 0.999
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}) :
        if(logs.get('acc') is not None and logs.get('acc') >= accuracy) :
            print('\nReached 99.9% accuracy so cancelling training!')
            self.model.stop_training = True

# Create a convolutional model
def convolutional_model():
  model = tf.keras.models.Sequential(
      [
          tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
          tf.keras.layers.MaxPooling2D(2, 2),
          tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
          tf.keras.layers.MaxPooling2D(2, 2),

          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(512, activation = 'relu'),
          tf.keras.layers.Dense(10, activation = 'softmax')
      ]
  )
  model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['Accuracy'])
  return model



model = convolutional_model()
# Count the number of parameters in the model
model_params = model.count_params()

# Print the number of parameters
assert model_params < 1000000, (
    f'Your model has {model_params:,} params. For successful grading, please keep it '
    f'under 1,000,000 by reducing the number of units in your Conv2D and/or Dense layers.'
)
#compile the model
callbacks = myCallback()
#increase the number of epochs to 20 for better results
history = model.fit(training_images, training_labels, epochs = 10, callbacks=[callbacks])
#evalute model
print(model.evaluate(test_images, test_labels))
predictions = model.predict(test_images)
print(predictions[0])
print(test_images[0])


