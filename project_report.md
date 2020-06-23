# Machine Learning Engineer Nanodegree
## **Capstone Project: Dog Breed Classifier**
Francesco Paolo Albano

22/06/2020

## I. Definition
### Project Overview
The goal of this project is to develop a model that given an image, it will predict the dog's breed. This model is mounted in a inference pipeline that recognizes if in the photo there is a human face or a dog. If a dog is detected in the image, the dog's breed model is invocated. If a human is detected, it will provide an estimate of the dog breed that is most resembling. If neither is detected in the image, the pipeline will return a message of no prediction. This is a classic computer vision supervised machine learnin problem, targeted to multi class image classification for the dog's breed. The datasets, provided by Udacity, are two, the first is a dataset of dog images divided by breed, the second is a dataset of human faces (the famous flw dataset<sup>1</sup>).
All the project was executed on a notebook equipped with a CPU i7-7700HQ and a GPU Nvidia Quadro M1200.

### Problem Statement
The inference pipeline needs two different models to solve the problem *Classify the dog's breed from an image sent by a user*. So we need of:
* Model for human face identification
* Model for dog identification
* Model for dog's breed classification

#### Human face identication
To solve this task, we can simply use a pretrained model based on Haar Cascade Classifier. Haar Cascade is a machine learning object detection algorithm used to identify objects in an image or video and based on the concept of ​​ features proposed by Paul Viola and Michael Jones in their paper "Rapid Object Detection using a Boosted Cascade of Simple Features"<sup>2</sup> in 2001. We have choosen the implementation from OpenCV package<sup>3</sup>. 

#### Model for dog identification
To identify if an image represents a dog, the pretrained VGG16 will be used. We have choosen the pretrained model from torchvision package <sup>4</sup>. This model is pretrained on ImageNet, a very large, very popular dataset used for image classification and other vision tasks. Looking at the ImageNet dictionary we can find that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary<sup>5</sup> keys 151-268.
 

#### Breed dog classification
To solve this task, we need a CNN model that is able to distinguish the breed of dog. The first approach is to build and train a CNN model from scratch, but the accuracy obtained was 24% which is resonable because of very simple architecture. Then we have using the tranfer learning to increasy accuracy and speed up the time to training. Using transfer learning on the Resnet18 model, from torchvision package, we are able to achieve 77% with very little effort and few training epochs.
The choice fell to the ResNet18<sup>6</sup> because have very good performance despite a not very complex model architecture.

### Metrics
This problem is a supervised classification task. The data is split in train, validation and test. Also there is not requirement to limit false positive or false negavite. So **Accuracy** is a good metric for this king of problem.
<center><img src="capstone_proposal_images/accuracy.svg"></center>
Moreover, because the dataset is a little unbalanced, the **F1 score** would produce a more realistic result. The F1 score (or F-score) is the harmonic mean of precion and recall.
<center><img src="capstone_proposal_images/precision.svg"><img src="capstone_proposal_images/recall.svg"></center>
<center><img src="capstone_proposal_images/f1score.svg"></center>

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?


### Bibliographies
1) LFW dataset: http://vis-www.cs.umass.edu/lfw/
2) Rapid Object Detection using a Boosted Cascade of Simple Features:  http://web.iitd.ac.in/~sumeet/viola-cvpr-01.pdf
3) Haar cascade object detection works: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
4) TorchVision pretrained models: https://pytorch.org/docs/stable/torchvision/models.html
5) ImageNet dictionary: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
6) ResNet: https://arxiv.org/pdf/1512.03385.pdf