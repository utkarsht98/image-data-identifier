# image-data-identifier
Extract useful info from images

# Model Training
1. To train the model, I used the concept of transfer learning.
2. Used the YOLO v3 darknet model and custom trained it specifically for face and documents like ADHAAR, PAN etc.
3. Refer the official darknet documentation - [Github link](https://github.com/AlexeyAB/darknet#requirements)
4. By tuning the hyperparameters and training for multiple epochs, finally obtained a decent object detector for official documents.
5. Searched the web for training images and labelled them using labelling tool for YOLO v3 training.

# Steps to use use the trained model

1. Clone this repo to your local machine
