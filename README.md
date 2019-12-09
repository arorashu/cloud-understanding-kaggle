# Understanding Clouds Kaggle Competition
Kaggle competition: Understanding Clouds from Satellite Images

## TODO
- checkout Snorkel - Stanford
	deals with not very good data labels
- look at various types of kernels
- look at various kaggle kernels to understand significant methods / ideas


## Members
- Shubham Arora
- Michael Brunsman

## Project Mission
The goal of this project is to create an image processing program that can identify different kinds of clouds (fish, flower, gravel, and sugar). This product will be used to predict weather in a region and to see the change in clouds due to climate change. Unlike most cloud image processing products, this one will be able to identify types of clouds in a region rather than just cloud density.
## Link to Competition
The competition web page is provided [Here](https://www.kaggle.com/c/understanding_cloud_organization/data).
## Target User(s)
- Climate Change Researchers
- Meteorologists
## User Stories
- I, the researcher, should be able to use this program to see the effects of climate change in a region.
- I, the researcher, should be able to use this program to see what types of clouds are in a region.
- I, the researcher/meteorologists, should be able to use this program to predict weather in a region.
## MVP
Our MVP’s bare bone features would be able to take in a satellite image (or a folder full of such images), analyze each one using an image processing API (such as scikit-image), and then return a sentiment detailing the density of each cloud type.
## Testing/Design
To test the product, the competition provides satellite (provided by NASA). The product should be able to analyze each of these images and identify the types of clouds present in the image and the density of these cloud types.

