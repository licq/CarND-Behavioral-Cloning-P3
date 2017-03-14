#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I use the nvidia model as start, it consists of 5 convolution neural network and 4 fully connected layers.
 
- The first cnn layer with 5x5 filter sizes and depths 24 and strides 2x2 (model.py line 89).
- The second cnn layer with 5x5 filter sizes and depths 32 and strides 2x2 (model.py line 90).
- The third cnn layer with 5x5 filter sizes and depths 48 and strides 2x2 (model.py line 91).
- The forth cnn layer with 3x3 filter sizes and depths 64 (model.py line 92).
- The fifth cnn layer with 3x3 filter sizes and depths 64 (model.py line 93).

- After the cnn layers, I use flatten layer. (model.py line 94)
- After that, there are 4 fully connected layers.
- The first dense layer has depth 100. (model.py line 96)
- The second dense layer has depth 50. (model.py line 97)
- The third dense layer has depth 10. (model.py line 98)
- The forth dense layer has depth 1. (model.py line 99)


- The model includes RELU layers to introduce nonlinearity in each cnn layer and all dense layers except the last one, and the data is normalized in the model using a Keras lambda layer (code line 87). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 95). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 204). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 180-182).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, reverse run data ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use existing models like lenet, nvidia, comma_ai.

My first step was to use a convolution neural network model similar to the lenet model.  I thought this model might be appropriate because I am familiar with it and it has very short training time.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high squared error on the training set but a high mean squared error on the validation set. This implied that the model was underfitting.
And my car always leave the road.

I think maybe lenet is too navie for the driving problem, so I changed my model to nvidia model.
The nvidia model has a very low training error and high validation error, so it is overfitting.

To combat the overfitting, I modified the model add a dropout layer with probability 0.5 between the flatten layer and the first dense layer so that the validation error become low. 

Then I tried to resize image to 66*200 before training, but it doesn't make the performance better.

And I also tried the comma_ai model, It worked worse than the nvidia model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. to improve the driving behavior in these cases, I trained some other data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

And after I finished track1, I added data for track2. I use the same model and same training process, now the car can complete track1 and track2 without leaving the road. The car behaves some strange behavior like change the steering rapidly in track1 and it cross the center line in track2 sometimes.
So data is more important.

####2. Final Model Architecture

The final model architecture (model.py lines 86-99) 

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the udacity recorded data first. And after some experiments, I added my own track1 training data, track2 training data. I recorded 2 times each for one normal direction and one reverse direction

To augment the data sat, I also flipped images and angles thinking that this would make more training data and make the model generalize better. 

After the collection process, I had 102840 number of data points. 

I tried some preprocess like resize and change color to hsv. It doesn't make the result better.

I also tried use random data generator to output random images to the train process, it makes the result worse.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.

####4. Transfer Learning

I tried to use transfer learning. I trained the model for 10 epochs. After driving in the simulator, I collect some data where perform not very well and load the pretrained model and finetine it on the new data.

