# Behavioral Cloning Project Writeup 

This README documents the approach and model parameters used in the Behavioral Cloning Project. 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


## 1. Required files 

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

### Run files
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Training file
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Data Collection and Training Strategy

### Camera data collection 

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer to the center of the lane should it veer off. These images show what a recovery looks like starting from ... :

The dataset was then augmented by flipping the camera images and their steering angle measurements. For example, here is an image that has then been flipped:


### Processing techniques

The post processing techniques were employed to improve the operational efficiency as well as the accuracy of the dataset collected. 

1. Cropping: The images were cropped (using the Keras Cropping 2D function) so only the image information pertaining to lanes were fed into the neural network. This is faster on Keras and also results in less data, improving the training speed while removing irrelevant information which could affect the neural net's weights at the same time. 

2. Steering angle compensation: Camera data from all three cameras were included. However, as the left and right cameras have a perception offset for the same steering angle, a correction factor was introduced to compensate for this offset. The formulas are: 
* Left steering angle = center steering angle + offset 
* Right steering angle = center steering angle - offset 

3. Normalization: The image data values have been normalized between 0 and 1, similar to the image pre-processing method in the Traffic Sign Classifier Assignment. 

## Model Architecture 

This model is modified from a model introduced in a paper written by Nvidia's engineers on behavioral cloning, titled "End to End Learning for Self-Driving cars" 

The final model architecture below is as follows: 

| Layer         		|     Description	        	| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 	45x320x3 Grayscale image  			|
| Convolution 5x5     	| 5x5 kernel with [2,2] stride, outputs 24 layers 	| 
| RELU 			| RELU	Activation function				|
| Convolution 5x5   	| 5x5 kernel with [2,2] stride, outputs 36 layers	| 
| RELU    		| RELU Activation Function				|
| Convolution 3x3	| 3x3 kernel, outputs 64 layers				|
| RELU			| RELU Activation function				|
| Convolution 3x3 	| 3x3 kernel, outputs 64 layers				|
| Fully Connected Layer | Outputs 100 classes					|
| Fully Connected Layer | outputs 50 classes					|
| Fully Connected Layer | Outputs 10 Classes					|
| Fully Connected Layer | Outputs Steering angle				|

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 25).

To prevent over-fitting, the use of dropout values was introduced at the 2nd and 3rd last connected layer. Dropout values tested were 0.2, 0.5 and 0.6. However, during these scenarios, the model was unable to round sharp bends, especially those where lane markers had been absent on one side of the road. 

As such, it was concluded that the final model architecture performed better without the use of dropouts. 



