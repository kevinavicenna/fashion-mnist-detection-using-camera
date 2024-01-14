import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# load model
model = load_model("C:/Users/kevin/Downloads/fashion_mnist_model.h5")

# labeling
class_labels = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
                5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

# Function to process images from the camera and classify them
def classify_from_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        img = cv2.resize(frame, (28, 28))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img_array = np.expand_dims(img_gray, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Classify the image using the model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        # Display the classification result on the screen
        cv2.putText(frame, class_labels[predicted_class], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Clothing Classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

classify_from_camera()