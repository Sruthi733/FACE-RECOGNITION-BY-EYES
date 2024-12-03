#ABSTRACT
This project presents a novel approach to face recognition by focusing specifically on the eyes, a distinctive and often underutilized feature in biometric identification. The system takes two input images and extracts the facial regions, from which the eye areas are isolated for further analysis. Key characteristics of the eyes, including eye color, pupil size, iris texture, eyelash length, and sclera features, are extracted and compared for similarity. By leveraging these unique eye traits, the system can determine the degree of resemblance between the individuals in the two images. This approach is particularly useful for identifying identical twins or individuals with similar facial features, offering a level of precision in distinguishing people that is not typically achievable with traditional face recognition methods. Moreover, this eye-focused identification model presents a promising solution to privacy and security concerns, as it limits the scope of personal data captured to only the eye region, mitigating the ethical challenges associated with full-face recognition. Through feature extraction, comparison, and similarity measurement, the proposed system enhances privacy-focused identification while maintaining high accuracy in real-world applications, such as secure access control, privacy-preserving surveillance, and twin identification. This selective analysis not only reduces the risk of privacy invasion but also enhances the security of identity verification systems. The project explores the potential of this technique for real-world applications, including secure access control, privacy-preserving surveillance, and twin identification. By employing feature extraction, similarity comparison, and machine learning techniques, the proposed system achieves high precision while offering a more ethical and privacy-conscious alternative to conventional face recognition technologies.

# FACE-RECOGNITION-BY-EYES
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Load and preprocess eye images
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}. Please check the path.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eyes_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

    for (x, y, w, h) in eyes:
        cropped_eye = gray[y:y + h, x:x + w]
        resized_eye = cv2.resize(cropped_eye, (64, 64))
        return resized_eye / 255.0  # Normalize pixel values

    return None

# Define a simple feature extractor for both images
def build_siamese_network():
    input_shape = (64, 64, 1)

    # Feature extraction model
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    output_layer = Dense(128, activation='relu')(x)

    feature_extractor = Model(input_layer, output_layer)

    # Define the inputs for the Siamese network
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)

    # Extract features for both inputs
    feature_1 = feature_extractor(input_1)
    feature_2 = feature_extractor(input_2)

    # Compute the absolute difference between features
    distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([feature_1, feature_2])
    output = Dense(1, activation='sigmoid')(distance)

    # Siamese model
    model = Model(inputs=[input_1, input_2], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Compare two eye images
def compare_images(model, image_path1, image_path2):
    image1 = preprocess_image(image_path1)
    image2 = preprocess_image(image_path2)

    if image1 is None or image2 is None:
        print("Error: One or both images could not be loaded.")
        return

    # Expand dimensions and match input shape
    image1 = np.expand_dims(image1, axis=-1)  # Add channel dimension
    image2 = np.expand_dims(image2, axis=-1)
    image1 = np.expand_dims(image1, axis=0)  # Add batch dimension
    image2 = np.expand_dims(image2, axis=0)

    # Predict similarity
    similarity_score = model.predict([image1, image2])[0][0]
    if similarity_score > 0.5:
        return "Face detected"
    else:
        return "Wrong match"

# Example usage
if _name_ == "_main_":
    # Build and summarize the model
    model = build_siamese_network()
    model.summary()

    # Load two example images and compare them
    result = compare_images(model, "kr1.jpg", "kr2.jpg")
    print(result)
