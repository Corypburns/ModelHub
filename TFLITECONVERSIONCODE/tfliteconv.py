import tensorflow as tf

# Load pretrained InceptionV3 Keras model
model = tf.keras.applications.InceptionV3(weights='imagenet')

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to disk
with open('inceptionv3.tflite', 'wb') as f:
    f.write(tflite_model)
