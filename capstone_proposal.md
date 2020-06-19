# Machine Learning Engineer Nanodegree
## Capstone Proposal
Francesco Paolo Albano 19/06/2020

[Project link](https://github.com/PaoloAlbano/project-dog-classification-udacitynanodegree)

<center><img width=250 src="capstone_proposal_images/dog_study.jpg"></center>

## Domain Background
The dog breed classification from an image is very challeging problem ever for a human eye. This is a multi class classification problem where use supervised learning. With this project we will build a pipeline to identify from a single image, if a dog is present in image, and the it breed. The dataset of dog images is downloaded from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). The dataset contains 8351 images of dog, divided into 133 dog breed. 

## Problem Statement
The goal of this project is to use the Convolutional Neural Networks and the Transfer Learning to create a pipeline that classify the dog breeds. This pipeline will be insert in a web app where user can take a picture and obtain prediction from this pipeline. 

The pipeline is formed by:
- Human face detector that check if is present a human face in the image. 
- Dog detector that check if is present a dog in the image.
- Breed dog detector that predict the dog breed.

We want to achieve at least of 70% of accuracy on dog breed prediction. We will user the transfer learning to build a CNN model with a good accuracy and less training time compared to create and traing a complex CNN from scratch.

## Datasets and Inputs
For this project we have 2 datasets provided from Udacity. 

### Dog Dataset
This dataset, downloaded from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip), contains 8351 images of dog, divided into 133 dog breed. 
This dataset is already splitted in:
- *train*: 6680 images. (**80%**)
- *validation*: 835 images. (**10%**)
- *test*: 836 images. (**10%**)

In this histogram we can see the distribution of trainin images among the various dog breeds. With a mean of 50.23 images for dog breed. 

<center><img src="capstone_proposal_images/histogram_dob_breeds.png"></center>

We can see a small imbalancing between dog breed, we will try to reduce the proble using the data augmentation on train images using the (Transforms)[https://pytorch.org/docs/stable/torchvision/transforms.html] of library torchvision.

## Human Dataset
This dataset, downloaded from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). contains 13234 human images. All images have a shape of 250x250 pixel. We will use this dataset only for test a pre-built model for human face identifier.

## Solution Statement
To perform the classification 

## Benchmark Model

## Evalutaion Metrics

## Project Design

## Presentation

