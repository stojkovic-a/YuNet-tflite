import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(
    "./face_detection_yunet_2023mar.pb"
)
tflite_model = converter.convert()

# Save the model.
with open("tflite/model.tflite", "wb") as f:
    f.write(tflite_model)
