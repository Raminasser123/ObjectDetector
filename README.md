# AI-Based Security Check Object Detection

![Project Cover](https://github.com/Raminasser123/ObjectDetector/blob/main/images/cover.png)

## Introduction
This project utilizes a deep learning model to identify dangerous objects in security checks at airports. The script, developed in a Google Colab environment, accesses and processes data stored on Google Drive. It relies on the power of convolutional neural networks (CNNs), specifically a VGG16 model variant, to classify objects in X-ray images.

## Setup and Installation
### Prerequisites
- Google Colab account
- Access to Google Drive with required datasets

### Installation Steps
1. Clone or download this repository.
2. Upload the script to your Google Colab environment.
3. Ensure the data required by the script is available in your connected Google Drive.

## Usage
To run the script:
1. Open the script in Google Colab.
2. Run each cell sequentially to:
   - Mount your Google Drive.
   - Import the following libraries:
     - `numpy` for complex mathematical operations on arrays/matrices
     - `matplotlib.pyplot` for graphs and plots
     - `os` for operating system interactions
     - `tensorflow.keras` and its submodules for neural network implementation
     - `keras.layers` and `keras.models` for building the model architecture
     - `keras.callbacks` for implementing custom callbacks
     - `tensorflow.keras.preprocessing.image` for data augmentation and labeling
3. Load and preprocess the dataset.
4. Build and train the deep learning model.
5. Observe the output for model performance and object detection results.

## Model Details
The model uses a convolutional neural network (CNN) architecture, which is designed for processing data with a grid-like topology, such as images. Here is the architecture diagram of the model used:

![Model Architecture](https://github.com/Raminasser123/ObjectDetector/blob/main/images/image2.png)

Performance metrics are crucial for evaluating the effectiveness of the model. Below is a chart showing the accuracy of the model with different optimizers:

![Optimizer Performance](https://github.com/Raminasser123/ObjectDetector/blob/main/images/image3.png)

The following image is an example of the model's output, demonstrating the detection of a dangerous object in a scanned bag:

![Detection Example](https://github.com/Raminasser123/ObjectDetector/blob/main/images/image.png)

## Acknowledgments
Special thanks to the creators of the datasets and the developers of the TensorFlow and Keras libraries, which made this project possible.

## Contributing
We welcome contributions to this project. If you have suggestions or improvements, please fork the repository and submit a pull request.
