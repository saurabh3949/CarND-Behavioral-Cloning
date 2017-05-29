# Behavioral Cloning Project
In this project, I trained a Convolutional Neural Network which learns to predict steering angles for autonomous driving.

The goals of this are the following:
* Use the Udacity Car driving simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road  

##### Output video
[![IMAGE ALT TEXT](http://img.youtube.com/vi/dffvR-SMHEI/0.jpg)](https://youtu.be/dffvR-SMHEI)

---
### Files Submitted

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `preprocess.py` for image preprocessing functions, data augmentation and data iterator
* `model.h5` containing a trained convolution neural network
* `video.mp4` containing autonomous driving output video
* `README.md` (this file) summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Network Architecture

My model is based on [NVIDIA's steering angle prediction model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

The model is defined in lines 20-32 of `model.py`.

I trained this network to minimize the mean-squared error between the steering command output by the network, and the steering angles logged by the simulator. The figure below shows the network architecture, which consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. The input image is split into YUV planes and passed to the network.

![Netwok architecture](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-768x1095.png)

The layers are summarized in the table below:

|Layer (type)    |             Output Shape     |         Params # |  
|:---|:---:|---:|
| Normalization (Lambda)     |      (None, 66, 200, 3)   |   0         |
| conv2d_1 (Conv2D)     |      (None, 31, 98, 24)   |   1824      |
| conv2d_2 (Conv2D)     |      (None, 14, 47, 36)   |   21636     |
| conv2d_3 (Conv2D)     |      (None, 5, 22, 48)    |   43248     |
| conv2d_4 (Conv2D)     |      (None, 3, 20, 64)    |   27712     |
| conv2d_5 (Conv2D)     |      (None, 1, 18, 64)    |   36928     |
| dropout_1 (Dropout)   |      (None, 1, 18, 64)    |   0         |
| flatten_1 (Flatten)   |      (None, 1152)         |   0         |
| dense_1 (Dense)       |      (None, 100)          |   115300    |
| dense_2 (Dense)       |      (None, 50)           |   5050      |
| dense_3 (Dense)       |      (None, 10)           |   510       |
| dense_4 (Dense)       |      (None, 1)            |   11 |       


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (`model.py line 27`).

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an ADAM optimizer, so the learning rate was not tuned manually (`model.py line 35`).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I collected the training data in the following manner:
- two laps of center lane driving
- two laps of center lane driving in reverse manner
- one lap of recovery driving from the sides
- one lap focusing on driving smoothly around curves

#### 5. Data Preprocessing and Augmentation

##### Preprocessing
- The images are cropped so that the model won’t be trained with the sky and the car front parts
- The images are resized to 66x200 (3 YUV channels) as per NVIDIA model
- The images are normalized (image data divided by 127.5 and subtracted 1.0) to avoid saturation and faster convergence.

#### Data Augmentation
I applied the following augmentation techniques in the images data iterator (python generator):
- For left image, steering angle is adjusted by +0.2
- For right image, steering angle is adjusted by -0.2
- Randomly flip image left/right
- Randomly altering image brightness (lighter or darker)

To introduce more randomness in the training set, these augmentation steps were applied with 50% randomness i.e. in each data iteration step, these augmentations would be applied with 50% probability. This technique helped the network generalize better.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

#### 6. Results
The model can drive the course smoothly without going off the track.  
Here's the [YouTube link](https://youtu.be/dffvR-SMHEI) for the Lake Track.
