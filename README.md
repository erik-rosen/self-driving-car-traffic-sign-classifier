# **Traffic Sign Recognition** 

This project aims to build a traffic sign classifier using convolutional neural net. The project is a part of Udacity's self driving car degree. In the project I achieved an F1 score of .964 on the Test set, consisting of images like the below:

![alt text][image2]

## Writeup


[//]: # (Image References)

[image1]: ./writeup_figures/training_set_classes.png
[image2]: ./writeup_figures/sample_images.png
[image3]: ./writeup_figures/synthetic_images.png "Synthetic Images"
[image4]: ./writeup_figures/synthetic_data_histogram.png "Balanced training set with synthetic data"
[image5]: ./writeup_figures/local_and_global_normalization.png "Local Normalization"
[image6]: ./writeup_figures/convnet_skip_connections.png "Implemented Architecture - image from http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf"
[image7]: ./writeup_figures/beware_snow.png "Beware Snow/Ice"
[image8]: ./writeup_figures/bicycles_crossing.png "Bicycle Crossing"
[image9]: ./writeup_figures/priority.png "Right-of-way at the next intersection"
[image10]: ./writeup_figures/misclassification.png "Misclassified images"
[image11]: ./writeup_figures/double_curve.png "Double curve"
[image12]: ./writeup_figures/general_caution.png "General Caution"
[image13]: ./writeup_figures/test_images.png "Test images from the web"
[image14]: ./writeup_figures/performance.png "Test images classification"

All code written for in this project can be found in: Traffic_Sign_Classifier.ipynb. 

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training data set. It is a histogram chart showing the occurence of each class in the training data set. From this we see that the data is very imbalanced in terms of occurences of different classes present. Training the classifier on an imbalanced dataset can lead to poor performance, as blind guessing of the most common class of the classifier can do a pretty good job to minimize the cost function, so we will need to handle this in the preprocessing steps.

![alt text][image1]

Visualizing some sample images of the training set below tells us that we will need to build a classifier that is able to deal with: 

* Variations in lighting conditions - both global lighting variations affecting the overall brightness of the image, as well as local lighting variations, such as shadows cast on the signs, glare, etc.
* Affine transformations, such as rotations, scale, and skew.
* Partial Occlusions

![alt text][image2]

### Design and Test a Model Architecture

As a first step, to ensure that there is class balance in the traning data, I generated additional synthetic data to ensure that each class in the training dataset had 2500 images. I did this by sampling from the existing images within each class, and applying a random rotation between -20 and 20 degrees, applying random translation between -2 to +2 pixels in the vertical and horizontal directions, as well as applying random shadow polygons to simulate variations in local lighting (using the Automold library: https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library).

See below for some example synthetic images:

![alt text][image3]

Below is the histogram of the resulting balanced training set (2500 images in each class):

![alt text][image4]

As a pre-processing step, convert from RGB to YUV color space, and then take the luma channel (brightness). We apply local normalization using this technique: http://bigwww.epfl.ch/sage/soft/localnormalization/ in order to make the resulting images less sensitive to lighting variations, resulting in the following output: 

![alt text][image5]

This ensures that the mean and variance of pixels within any given patch of in the image (the size of this patch is determined by the parameter values used) is approximately zero mean and of unit variance respectively.

I implemented the architecture described in this paper: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf, which contsists of the following layers:

![alt text][image6]

| Layer         		|     Description	        					                      | 
|:---------------------:|:---------------------------------------------:                      | 
| Input         		| 32x32x1 Locally normalized Luma channel image	                      | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x100 	                      |
| RELU					|												                      |
| Max pooling	      	| 2x2 stride,  outputs 14x14x100 				                      |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x200 	                      |
| RELU 		            |         								     	                      |
| Max pooling		    | 2x2 stride,  outputs 5x5x200	    			                      |
| Fully connected		| Input: flattened output from both max pooling sections, output: 43x1|
| Softmax				|         									                          |


To train the model, I used tensorflow's tf.train.AdamOptimizer and defined the cost function to be the cross entropy and set the EPOCHS to 10 and BATCH_SIZE to 32. I reduce the learning rate by 2/3rds if there is no improvement in the accuracy between epochs to ensure that there is convergence in the optimization. I could probably reduce the number of epochs used as validation accuracy does not increase much past the first few epochs and could lead to overfitting given that there is no regularization / dropout implemented in the model.

At first, applied the LeNet architecture with gave me 94% accuracy with the described preprocessing steps above. To improve that further, I chose to implement one of the top performing architectures for the traffic sign dataset described here: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf. The architecture is relatively simple with 2 convolution layers, and uses a feed forward technique that feeds the output of the initial convolution stage to the final, fully connected layer. Quoting the paper: "This allows the classifier to use, not just high-level features, which tend to be global, invariant, but with little precise details, but also lowlevel features, which tend to be more local, less invariant, and more accurately encode local motifs."

My final model results were:
* training set accuracy of .998
* validation set accuracy of .981
* test set accuracy of 0.964

Below is the precision, recall and F1 score for each class:

                                                         precision    recall  f1-score   support
                                             Ahead only      0.990     0.982     0.986       390
                                     Beware of ice/snow      0.913     0.560     0.694       150
                                      Bicycles crossing      0.796     1.000     0.887        90
                                             Bumpy road      0.992     0.975     0.983       120
                                      Children crossing      0.993     0.980     0.987       150
                            Dangerous curve to the left      0.923     1.000     0.960        60
                           Dangerous curve to the right      0.957     1.000     0.978        90
                                           Double curve      0.924     0.678     0.782        90
                    End of all speed and passing limits      0.984     1.000     0.992        60
                                      End of no passing      0.821     0.917     0.866        60
     End of no passing by vehicles over 3.5 metric tons      0.963     0.878     0.919        90
                            End of speed limit (80km/h)      1.000     0.853     0.921       150
                                        General caution      0.914     0.959     0.936       390
                                    Go straight or left      1.000     0.983     0.992        60
                                   Go straight or right      0.983     0.992     0.988       120
                                              Keep left      0.978     1.000     0.989        90
                                             Keep right      0.987     0.971     0.979       690
                                               No entry      1.000     0.969     0.984       360
                                             No passing      1.000     0.994     0.997       480
           No passing for vehicles over 3.5 metric tons      0.998     0.964     0.981       660
                                            No vehicles      0.986     1.000     0.993       210
                                            Pedestrians      0.938     1.000     0.968        60
                                          Priority road      0.964     0.980     0.972       690
                  Right-of-way at the next intersection      0.925     0.938     0.931       420
                              Road narrows on the right      0.906     0.967     0.935        90
                                              Road work      0.983     0.973     0.978       480
                                   Roundabout mandatory      0.919     0.878     0.898        90
                                          Slippery road      0.768     0.973     0.859       150
                                  Speed limit (100km/h)      0.964     0.953     0.959       450
                                  Speed limit (120km/h)      0.949     0.984     0.966       450
                                   Speed limit (20km/h)      0.964     0.900     0.931        60
                                   Speed limit (30km/h)      0.999     0.979     0.989       720
                                   Speed limit (50km/h)      0.938     0.996     0.966       750
                                   Speed limit (60km/h)      0.992     0.878     0.932       450
                                   Speed limit (70km/h)      0.983     0.982     0.983       660
                                   Speed limit (80km/h)      0.882     0.976     0.927       630
                                                   Stop      0.993     1.000     0.996       270
                                        Traffic signals      0.938     0.922     0.930       180
                                        Turn left ahead      0.975     0.992     0.983       120
                                       Turn right ahead      0.995     0.986     0.990       210
               Vehicles over 3.5 metric tons prohibited      1.000     0.993     0.997       150
                                  Wild animals crossing      0.996     0.981     0.989       270
                                                  Yield      1.000     0.996     0.998       720
     
                                            avg / total      0.966     0.964     0.964     12630

While performance is generally pretty good, the model does not perform very well on the following classes: 

* Beware of ice/snow
* Double curve

For both of these images, the recall is low, i.e. the classifier erroneously mistakes these classes for something else - in 56% of cases when a Beware of ice/snow sign is shown, the classifier will predict that it is something else. In 68% of cases, when a double curve is shown, the classifier will predict something else. 

Looking at the confusion matrix, we see that the classifier most often confuses the "Beware of ice/snow"-sign with "Bicycles crossing" (14.6% of cases) or "Right-of-way at the next intersection" (12% of cases), where as when the "Double curve" sign is shown it is commonly misclassified as a "General caution" sign (32% of cases).

Regarding misclassifications of "Beware of ice/snow", we note that "Beware of ice/snow", "Bicycles crossing" and "Right-of-way at the next intersection" all have similarly shaped sign board, and the symbols in the center of the signs are of somewhat of similar scale. 

![alt text][image7] 
![alt text][image8] 
![alt text][image9]

Looking at some of the misclassified images below, we can see that for the case of "Beware of ice/snow", poor lighting and blur makes it very challenging to tell apart the central motif of the sign - it ends up looking like a blob. This would explain the poorer performance in classifying this particular sign. It might be possible to improve recall for this by reducing the downsampling by removing the pooling (at the expense of increasing the amount of weights to train / size of the model).

![alt text][image10] 

In the case for the double curve sign misclassification, we note that the double curve and general caution signs are fairly similar as well, with similarly shaped sign boards, and central motifs which have similar scale and vertical orientation. 

![alt text][image11] 
![alt text][image12] 

As with the "Beware of ice and snow case", it might be possible to improve recall for "Double Curve" by reducing the downsampling by removing the pooling (at the expense of increasing the amount of weights to train / size of the model). This would allow the network to pick up more of the finer details in the central motifs (such as the bend in the double curve).
 

### Testing the Model on New Images

Here are ten German traffic signs that I found on the web:

![alt text][image13] 

These are images with very low noise, so I expect very good performance.

#### 2. Model predictions 

Here are the results of the predictions alongside with the softmax probabilities:

![alt text][image14] 

The accuracy of these 10 predictions is 100%, compared to 96.4% on the test set

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the images downloaded from the web, the model is 100% certain for every prediction, which is fair given the low noise in the images. 

What is perhaps more interesting is looking at the images in the test set (more representative of real conditions) that the classifier gets wrong:

![alt text][image10] 

Notably, the classifier seems overconfident in it's softmax probabilities - for the cases of the "keep right" signs, the "beware of ice/snow" signs and the top right "speed limit 60" sign, it is nearly completely certain in its erroneous predictions. This could indicate that the model is overfitting and needs regularization. 

The model does not make use of the color in the image, which would probably help correctly classify the "keep right" signs. 

The Traffic signals sign is heavily rotated. By introducing a larger degree of random rotation in the generation of synthetic data, we could potentially correctly classify this sign.

Whereas it is unclear why the model gets the bottom middle "Speed limit 20" and top right "Speed limit 60" wrong, the rest of the cases are fairly difficult even for a human to correctly classify due to poor lighting conditions, blur and occlusion.

