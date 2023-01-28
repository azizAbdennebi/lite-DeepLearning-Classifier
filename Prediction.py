import tensorflow as tf
import numpy as np
from PIL import Image


loaded_model = tf.keras.models.load_model("saved_model/Classification")


img = Image.open("auto.jpg")

# Resize and normalize the image
(img_rows, img_cols)=32,32
img = img.resize((img_rows, img_cols))
img_array = np.array(img)
img_array = img_array.astype("float32") / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make a prediction
predictions = loaded_model.predict(img_array)


# Get the predicted class
predictions = np.argmax(predictions)
classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
predicted_class=classes[predictions]
print("Predicted class:", predicted_class)
