** The documentation process are still in progress 

# ClimateHack-UCL
This is our attempt in creating a Machine Learning model that is able to predict future satellite imagery

Shown below is the result of our models:

<img src="https://user-images.githubusercontent.com/82151839/160614308-f2f995c3-da3a-4d04-94c9-38571dbe928f.gif" width="600" height="200">
&emsp;&emsp;&emsp; Ground Truth &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;  PredRNN &emsp;&emsp;&emsp;&emsp; PredRNN + Optical Flow

&emsp;

This model has allowed our team to reach a score of ~0.78 in the competition

<img width="500" alt="Screenshot 2022-03-29 at 16 50 36" src="https://user-images.githubusercontent.com/82151839/160652958-3c540f4e-c446-4606-a64b-83f1fbb3e111.png">


## Setup
**In Progress

## Data Preprocessing
Since the challenge of this competition is to predict the centre of the next 24 images (dimension: 64x64) from the given 12 images (Dimension: 128x128),
all of the input data is of size 64x64. The input data consists of 9 channels, in which:
  - 1 Channel : Centre-cropping
  - 1 Channel : Max-pooling
  - 1 Channel : Mean-pooling
  - 4 Channels : Space-to-depth
  - 2 Channels : Coordinates (x-osgb & y-osgb)

The motivation behind this preprocessing method is attained from [MetNet: A Neural Weather Model for Precipitation Forecasting](https://arxiv.org/abs/2003.12140)

## Model
There are two models that has been put to the final presentation.
### 1.PredRNN

<img width="400" alt="Screenshot 2022-03-29 at 16 56 07" src="https://user-images.githubusercontent.com/82151839/160654086-0d2ccd14-9457-475a-9459-f60a9eab63dc.png">

Our base model consists of PredRNN layer. The PredRNN layer received one input at a time (preprocessed data at one timestep), and its goal is to predict the satellite imagery of the next timestep.

The PredRNN layer is simply an improved version of ConvLSTM, where it introduces a new hidden states (which is called memory states) with a zig-zag flow. With an addition of memory states, the model is able to capture more complex information that otherwise won't be able to be captured by a vanilla ConvLSTM model.

There are also a few tricks employed during the training
- Scheduled Sampling
- Memory Decoupling

For further information on PredRNN, please refer [PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning](https://arxiv.org/abs/2103.09504)

**Some of the tricks mentioned for training are also attained from the aforementioned paper

### 2.PredRNN + Optical Flow + Axial Attention

<img width="500" alt="Screenshot 2022-03-29 at 16 54 25" src="https://user-images.githubusercontent.com/82151839/160653801-f5395cf0-c477-4ef4-920b-2784654ec88b.png">

We then extend our basic model to incorporate optical flow into our model. For the optical flow, we calculate the optical flow between all the 12 input images, which is then fed into convolutional layers with the goal to capture features that might be useful to the model. 

After the convolution layer, the optical flow features are then concatenated with the output of the PredRNN before being passed through axial attention layers. The purpose of axial attention layers are to focus on the important information that are useful to the model.

The tricks employed in the first model were also used for this model.

## Loss Function
For the loss function, we are using MS-SSIM loss. One trick that we employ during the training of our model is:

### Loss Adjustment

In the pipeline of our model, there are 36 set of images (12 for the inputs, 24 for the target). We designed our loss function such that initially, it will capture the loss of all the 36 images and then slowly shift our loss function so that it only capture the loss of the last 24 images.

Given the structure of our model where it predicts the future frames iteratively, we want our model to be able to build a good backbone in predicting the future frames. And to have a good backbone, it is essential for the earlier layers to have a good representation of the hidden states, hence why we believe incorporating the 12 input images into the loss function might be a good idea especially in the earlier iteration.

We then slowly shift the loss function (by adjusting the weight) to only focus on the last 24 images in the pipeline since our goal is to predict the next 24 frames of satellite imagery.

## Results and Analysis
Shown below are the training loss for both model:

<img width="400" src="https://user-images.githubusercontent.com/82151839/160659421-391536f6-3cf5-431a-8845-8c05fc3e83a7.jpg">

From the graph above, it can be seen that the performance of the updated model does not vary that much from the first model. We believe the reason could stem from the inductive bias that we introduce when processing the optical flow. The inductive bias that we indtroduce (convolutional layer) does not able to capture additional features that is useful for the model to learn.

we can see that our model (both Model 1 and Model 2) has done a decent job in predicting the satellite image at the first half of the video. However, at the second half of the videos, both models fail to predict the satellite image (in fact, our predicted images seem not to have any sort of movement in the cloud) effectively.

## Future Improvement
We believe that incorporating optical flow into the model is beneficial if done right. Hence, stemming from our previous analysis, instead of processing the optical flow features via convolutional layer, what if we let the data speak by itself, in other words, let the data determine what bias is good for the model.

The way this can be achieved is by doing cross attention between the optical flow and the output of PredRNN. This is then passed through the axial attention layer before we receive the next frame of the satellite imagery.

Shown below is the architecture of the updated model:


