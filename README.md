# Image Data Identifier
Extract useful info from images. In this project, if a person is holding an official government identification in the given image, then the algorithm extracts the person face and the document and classifies the given document into PAN card or ADHAAR.

# Model Training
1. To train the model, I used the concept of transfer learning.
3. Used the YOLO v3 darknet model and custom trained it specifically for face and documents like ADHAAR, PAN etc.
4. Refer the official darknet documentation - [Github link](https://github.com/AlexeyAB/darknet#requirements). The darknet code was not uploaded to this repo due to it's large size contraint.
5. By tuning the hyperparameters and training for multiple epochs, finally obtained a decent object detector for official documents.
6. Searched the web for training images and labelled them using labelling tool for YOLO v3 training.

# Steps to use use the trained model

1. Clone this repo to your local machine. Intall the dependencies using "requirement.txt" file.
2. Open the file "image_extract.py".
3. Edit line 119 to add absolute input image path that you want to input for testing. (Command line argument adding soon...)
4. Edit line 132 to add the absolute path of the ".cfg" file in "YOLO config" folder.
5. Edit line 133 to add the absolute path of the ".weights" file present in "cutom_weights" folder.
6. Run the "image_extract.py" module. Your extracted images will be saved in "updated_images" folder.
