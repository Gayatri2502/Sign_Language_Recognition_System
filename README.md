# Hand Gesture Recognition

## Project Overview
This project focuses on hand gesture recognition using computer vision and deep learning techniques. The primary objective is to capture hand images, preprocess them, and classify gestures using a trained deep learning model. The system can be useful in various real-world applications such as sign language recognition, touchless interfaces, and accessibility tools for differently-abled individuals.

## Project Structure
```
Internship Project
│── Data/                 # Contains captured hand gesture images
│── Model/                # Contains trained deep learning model & labels
│── dataCollection.py     # Collects hand gesture dataset
│── test.py               # Classifies hand gestures using the trained model
```

## Tools & Libraries Used
### 1. OpenCV (`cv2`)
   - Used for image processing and handling webcam input.
   - Helps in real-time detection, cropping, and resizing of hand images.

### 2. `cvzone.HandTrackingModule`
   - Provides an easy-to-use API for hand detection.
   - Identifies the bounding box of the detected hand, making cropping and processing easier.

### 3. `cvzone.ClassificationModule`
   - Used for loading and applying a pre-trained deep learning model.
   - Helps in predicting the gesture label from the processed hand image.

### 4. NumPy (`numpy`)
   - Used for numerical operations like image resizing calculations.
   - Helps in managing matrices and image transformations efficiently.

### 5. TensorFlow/Keras
   - Provides a framework for training and running deep learning models.
   - The pre-trained model is stored in `Model/keras_model.h5`, which is used for classification.

## Installation
To install the necessary dependencies for this project, run the following commands:
```
pip install cvzone==1.4.1
pip install mediapipe==0.8.3.1
```

## Dataset Collection Process
The `dataCollection.py` script is used to collect a dataset of hand gestures.

### How it Works:
1. The script accesses the webcam using OpenCV.
2. The `HandDetector` module detects hands in the video feed.
3. A bounding box is drawn around the detected hand.
4. The detected hand is cropped and resized to a fixed size of 300x300 pixels.
5. If the user presses the 's' key, the image is saved to the `Data/` folder.

### Why it is Used:
- Collecting data in a structured manner is essential for training an accurate machine learning model.
- The cropping and resizing ensure uniform image size, which helps in better model performance.

## Model Testing & Prediction
The `test.py` script is responsible for classifying hand gestures using the trained model.

### How it Works:
1. The script accesses the webcam and detects hands in the frame.
2. The detected hand is cropped and resized to 300x300 pixels.
3. The processed image is fed into the `Classifier` model.
4. The model predicts the gesture and displays the corresponding label on the screen.

### Why it is Used:
- Real-time hand gesture classification can be used in applications such as sign language recognition and virtual controls.
- The script leverages a pre-trained deep learning model to make accurate predictions.

## How to Run the Project
### Prerequisites
Ensure you have Python installed along with the required libraries:
```
pip install opencv-python cvzone numpy tensorflow
```

### Running the Data Collection Script
```
python dataCollection.py
```
- Press 's' to save captured images.

### Running the Gesture Classification Script
```
python test.py
```
- The script will classify hand gestures in real time and display predictions.

## Expected Outputs
1. `dataCollection.py`:
   - Displays the camera feed with detected hands.
   - Saves cropped images when 's' is pressed.
2. `test.py`:
   - Detects and classifies gestures.
   - Displays the recognized hand gesture label.

## References
- OpenCV documentation: https://docs.opencv.org/
- TensorFlow/Keras documentation: https://www.tensorflow.org/

