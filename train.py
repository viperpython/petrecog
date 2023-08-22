import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# Load the data
(train_data, validation_data, test_data), info = tfds.load('cats_vs_dogs',
                                                           split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                                           with_info=True,
                                                           as_supervised=True)
label_names = info.features['label'].names
# Define a function to display some sample images
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_data.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(label_names[label])
    plt.axis('off')
IMG_SIZE = 150
def format_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize the pixel values
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize to the desired size
    return image, label
batch_size = 32
train_data = train_data.map(format_image).shuffle(1000).batch(batch_size)
validation_data = validation_data.map(format_image).batch(batch_size)
test_data = test_data.map(format_image).batch(batch_size)
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy']) 
history = model.fit(train_data,
                    epochs=10,
                    validation_data=validation_data)
model.save('cats_vs_dogs.h5')
loaded_model = tf.keras.models.load_model('cats_vs_dogs.h5')
test_loss, test_accuracy = loaded_model.evaluate(test_data)
print('Test accuracy:', test_accuracy)
