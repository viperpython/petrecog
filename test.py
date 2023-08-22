import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# Load the data
(train_data, validation_data, test_data), info = tfds.load('cats_vs_dogs',
                                                           split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                                           with_info=True,
                                                           as_supervised=True)
loaded_model = tf.keras.models.load_model('cats_vs_dogs.h5')

IMG_SIZE = 150
def format_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize the pixel values
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize to the desired size
    return image, label
batch_size = 32
test_data = test_data.map(format_image).batch(batch_size)
loaded_model = tf.keras.models.load_model('cats_vs_dogs.h5')
# test_loss, test_accuracy = loaded_model.evaluate(test_data)
# print('Test accuracy:', test_accuracy)
for image, label in test_data.take(1):  # Loop through images and labels
    prediction = loaded_model.predict(image)
    
    plt.figure(figsize=(150, 150))
    correct=0
    wrong=0
    for i in range(len(prediction)):
        plt.subplot(7, 7, i + 1)
        plt.imshow(image[i])
        plt.xticks([])
        plt.yticks([])
        
        if prediction[i] > prediction.mean():
            predicted_class = 1
        else:
            predicted_class = 0
        
        color =''
        if predicted_class == label[i].numpy(): 
            color='blue'  
            correct+=1
        else:
            color='red'
            wrong+=1
        class_name = 'dog' if predicted_class == 1 else 'cat'
        plt.title(class_name, color=color)
    text = f"Correct: {correct}\nWrong: {wrong}\nAccuracy: {correct/32:.2f}"
    plt.text(0.5, -0.4, text, color='black', transform=plt.gca().transAxes,
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    # print("correctly predicted:",correct)
    # print("wrongly predicted:",wrong)
    # print("accuracy:",correct/32)
    plt.show()