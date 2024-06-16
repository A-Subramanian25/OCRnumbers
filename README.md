# OCRnumbers
**🔍 Project Overview**

In this project, I created a K-Nearest Neighbors (KNN) classifier to recognize handwritten digits from the MNIST dataset. The key steps include reading and processing the raw byte data, converting it to a format suitable for training and testing the KNN model, and evaluating the model's accuracy.

**🔧 Key Components**
**Reading the MNIST Data**
The images and labels are stored in byte format. We use NumPy to handle and process this data efficiently.

**Flattening Images**
Converting the 2D image data into 1D arrays to feed into our KNN algorithm.

**K-Nearest Neighbors (KNN) Algorithm**
Implementing the KNN algorithm to classify the images based on their pixel values.

**💡 Key Takeaways:**
  Efficient data handling and processing with NumPy.
  Implementation of KNN algorithm for image classification.
  Practical application in OCR and machine learning.

K- Nearest Neighbours algorithm is a simple classification algorithm.It is also known as lazy learner's algorithm. 
In this algorithm we find the distance between the given point and the training points and find the topmost K values which are near the testing point and find the majority of that collection.
   # K is mostly odd in order to help break ties
