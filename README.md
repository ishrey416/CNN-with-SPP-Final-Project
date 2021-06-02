# CNN-with-SPP-Final-Project

## Introduction
### Problem Definition
Now that neural networks are gaining more and more popularity for a variety of uses, the task of being able to train an entire network on several pieces of data seems more feasible than ever. New algorithms to transform and easily facilitate the classification of data are conceptualised everyday, and with these, limitations for which data is readily accepted as input are also suppressed. One such limitation is image dimension for image classification. If one were to attempt classification of images using a Convolutional Neural Network, one would have to ensure that the size of each image is exactly the same. This limitation would severely hinder the quantity of data that would be available, as well as limit the capabilities of the network to only acknowledge mono-sized images. This is the issue that we decided to tackle.
### Motivation
The problem of being limited to like-sized images as input is one that data scientists seem to simply accept. An entire collection of images of a certain class cannot be easily inputted into a neural network; instead, the images must all be painstakingly either cropped or warped to achieve the necessary size. In general, this means either cropping/warping every image to match the smallest one, or increasing the size of the smallest image and thus reducing its resolution. Inaccuracies in training the model would then prevail, and a less-accurate model overall would be generated. Our Spatial Pyramid Pooling implementation would be a way to skip these undesirable steps, and give more freedom for the image sizes that can be inputted by the user. 
### Contributions
For our project, we sought to implement an algorithm for Spatial Pyramid Pooling, a mechanism by which a Convolutional Neural Network that is being trained for image classification can be given an image dataset containing images of varying dimensions, which is a function that is usually not available. This would give it greater freedom for data input, and would lead to a better trained and thus more accurate model overall.
### Takeaways
We found that the SPP-incorporated CNN performs better than the base CNN on datasets of different sized images, and at least as good on datasets of fixed sized images.
## Description
For our various testing and comparison purposes, we implemented a total of three
algorithms: a no-frills implementation of a CNN trained on the CIFAR10 dataset (images all a consistent size), the same implementation of CNN trained on the CalTech101 dataset (images of varying sizes), an implementation of CNN that includes Spatial Pyramid Pooling, a method to allow the unadulterated input of multi-sized images to be trained on, trained on the CalTech101 dataset, and for exploration sake, an implementation of Spatial Pyramid Pooling CNN that was trained on the CIFAR10 dataset. Our primary comparison occurred between our vanilla, or no-frills, CNN model that was trained on the CalTech101 dataset and our SPP CNN model that was trained on the CalTech101 dataset.

 For standardization, all models were formulated to have: a 2D convolutional layer, followed by a pooling layer, followed by another 2D convolutional layer, followed by three fully-connected layers that perform a linear transformation to reduce the number of features. 
### CIFAR10 Baseline CNN
First, for the CIFAR10 implementation of our normal CNN model, we tried to approach it in a way where the only difference between it and the other models was the database being used to train. With this rule in place, we had to implement several transformations to each of the data inputs so that each model started off with equal footing. For instance, we encountered an issue early on in which our other models that used the CalTech101 database had to combat different numbers of channels among the input images. There were several images that only contained one channel, which meant that it was in grayscale, and the majority had three channels, which meant that they were in RGB, or full color. This resulted in transformation errors in the models, so to fix this, we made the choice to convert all images from every dataset for every model to grayscale. Performance would be generally worse, but at least each would be impacted in the same way and there would be no channel disparities.

 In every other way, this implementation was standard. Just like our other models, we had the same two convolutional layers, single pool layer, and three fully connected layers. Since we were training for the output to belong to one of ten classes, our final fully-connected layer’s output size was 10. During our forward pass, we applied the Rectified Linear Unit (ReLU)
function to each Linear transform as an activator layer, which is typical of a baseline CNN implementation.
### CalTech101 Baseline CNN
Our next model was again a generic CNN, but was instead trained on the CalTech101 dataset. Preprocessing for this dataset was notably different, as a normal CNN would not take differently-sized images as input, and thus, they would need to be resized so that training may occur. To facilitate this, we implemented the resizing of each image in the input to the transform method in our preprocessing. Each image was resized to 64 x 64, and the negative connotations of doing so were noted; there would obviously be some loss or distortion of information when the image size is changed, and we agreed that this would be the one area our SPP implementation would use to achieve higher accuracy. Lastly, the images were all normalized in regards to mean and standard deviation.

 For the implementation, it was essentially identical to that of the CNN model we designed for use with the CIFAR10 dataset. The only difference was that we ended our third fully-connected layer with a linear transformation to 101 features instead of 10, since CalTech101 comprised 101 classes.
### CalTech101 SPP CNN
The preprocessing of our implementation of a CNN with spatial pyramid pooling was exactly the same as in our CIFAR10 baseline CNN. This was done as a way to maintain consistency throughout the models and allow each to be trained starting with the same information, apart from unavoidable dataset differences.

 Our implementation of SPP was interesting. The role of the SPP layer is to act as an intermediary pooling layer between the convolutional layers and fully-connected layers, and essentially size input down to a fixed-size input. This is done after the convolutional layers and before the fully-connected layers because due to the way each of these layers operate, convolutional layers can take any sized input, while fully-connected layers can only take fixed-size layers. As a result, the SPP layer takes the output from the convolutional layer, reduces the dimensions of the feature maps of the input as necessary, and passes them to the fully-connected layers, which will now accept the dimensions of the transformed input. As can be seen in our method implementation, we use the dimensions of the previous convolutional layer, and the necessary output pool size, and the SPP method divides up the image into a certain
number of spatial bins in a way that does not depend on the size of the actual image; this is done proportionally so that the same number of features are extracted each time, and are then concatenated into a fixed-size output. The “pyramid” description of this method comes from the fact that this spatial bin extraction occurs
multiple times per pooling; in our implementation, we first extracted four bins, then two bins, then one bin, hence the “pyramid” (Figure 1).

 One issue we ran into was that our
data loader for the CalTech101 dataset did
not accept images of different sizes for
testing, so to facilitate this, we had to
manually load each image and its label from the dataset one by one, in a for-loop. This did severely hinder efficiency and took longer than standard loading would have taken, but did not have any noticeable effect on the accuracy of the model.
### CIFAR10 SPP CNN
Our implementation of this model was by and large the same as the one for the CalTech101 dataset. Performance-wise, we did not expect any interesting results; since CIFAR10 contains images that are all the same size, the use of an SPP layer was rendered redundant. As a way to explore the power of SPP, we started by using [4, 2, 1] bin pooling layers, like our CalTech101 implementation, but then expanded to one additional layer, [8, 4, 2, 1]. To our surprise, this added to the accuracy of the model.
### Training
All of our models that trained using the CIFAR10 dataset were trained on 256 batches, and every other model was trained on 8 batches; this was done to maximize performance. Each model used the Cross Entropy Loss function and the Adam stochastic gradient descent optimizer, and every model was trained for five epochs. These were kept constant between the base CNN and the SPP CNN so as not to add any confounding variables to our experiment.
  
## Evaluation
For our testing, we used two main datasets to facilitate the training of our models. We were primarily interested in the use of the Caltech101 image dataset, since it was one of the few datasets that contained images that were different sizes, which was an attribute that was necessary for testing the functionality of our SPP algorithm. We expected our unaltered CNN model to fare much worse in being trained on this specific dataset, as this model required consistent image sizes, and each image would have to be transformed prior to being inputted.

 In addition, in order to have a metric by which to measure how SPP would perform assuming that it was given a dataset where every image had the same dimensions, the second dataset we used was the classic CIFAR10 dataset. This was to test whether SPP could outperform our vanilla CNN model on a dataset with multiple-sized images, but if it could also hold its own on a dataset with images of all the same size, a characteristic that a vanilla CNN model would typically excel with.

 To measure our models’ accuracy, we created test sets from each dataset, where the corresponding correct class was known. We then ran our models with the test images, and simply measured how many of the test inputs it correctly classified.

 Our results largely fit our expectations of the superiority of our SPP CNN model compared to a baseline CNN model. When using the CalTech101 dataset, the vanilla CNN finished with an accuracy of 47%. This was low, which was not surprising, as many transformations had to have been applied to the inputs to make it viable with the CNN model, and this definitely hindered the model’s ability to recognize features. Since our SPP CNN model looked to alleviate this specific problem, we anticipated that it would fare better, and it did. This improved model got an accuracy of 57%.

 We also tested both the baseline CNN and SPP CNN on the CIFAR10 dataset, and unsurprisingly, the accuracies were exactly the same. Both ended with an accuracy of 57%: the baseline CNN probably improved as a result of its increased ability to train on the CIFAR10 dataset over CalTech101, and the SPP CNN likely stayed the same because with fixed-size inputs, it was like the pooling layer’s ability to pull features from different-sized images was not even there.

 In our attempts to improve this model, however, we strove to improve our implementation of SPP. We decided to add another pooling layer, in which a spatial bin of
features of size eight is extracted from the input; this gave us a total pooling layer pipeline of [8, 4, 2, 1]. This improved our accuracy to 60% altogether.
## Conclusions
In conclusion, the SPP-incorporated CNN showed more improvements compared to the regular CNN on the Caltech101 dataset and on the CIFAR10 dataset. For the Caltech101 dataset, over the course of 5 epochs, the loss of the SPP CNN was consistently lower than that of the regular CNN, which resulted in a 10% improvement in accuracy over the regular CNN. This proves that transformations such as cropping or warping pale in comparison to the SPP-layer which gathers data of different dimensions to have a complete understanding of the image. However, such an improvement was not so clear on the CIFAR-10 dataset. From the results, we can say that the SPP CNN performs at least as well as the regular CNN, if not better.
## Bibliography
 He et. Al, Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition - https://arxiv.org/pdf/1406.4729.pdf
 
 Caltech101 Dataset - http://www.vision.caltech.edu/Image_Datasets/Caltech101/
 
 Ouyang et. al, Pedestrian-Synthesis-GAN: Generating Pedestrian Data in Real Scene and Beyond - https://arxiv.org/abs/1804.02047
 
 Colab file: https://drive.google.com/file/d/1U2FwkRdBWYESlJUZJVnmjcZuUbj_zB3a/view?usp=sharing
