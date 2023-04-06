Sign Language Interpreter Project Proposal

Data Science Domains: Computer Vision, Classification Machine Learning 

Allen Lau, Sumaiya Uddin, Shubham Khandale


I. Introduction 

II. Motivation & Problem Statement

III. Related Work

IV. Dataset & Features 

V. Methodology

To build the Sign Language Interpreter, the following methodology will be followed: data loading/cleaning, exploratory data analysis, preprocessing (image transformations like conversion to gray images and edge detection and feature scaling), train-test dataset splitting, dimensionality reduction, feature selection, modeling, evaluation, and interpretation. This section of the project proposal will discuss important steps in the methodology in closer detail. 

Preprocessing aims to improve the performance of the model and prevent issues like overfitting. Firstly, the images will be converted to grayscale, which simplifies the images reducing the computational requirements in the data science workflow. Additionally, the Sign Language Interpreter does not need to train on the color of an individual’s hand to classify the image. Next, feature scaling ensures that models that are affected by scale are not overly affected by high pixel values, thus resulting in a lower chance of overfitting. Lastly, edge detection will be used to further decrease the complexity of the images in an attempt to improve the performance of the interpreter and decrease the runtime of training. This is due to the fact that only the edges of the hand are needed to understand which letter is being signed. 

Due to the high dimensionality of the image dataset, dimensionality reduction and feature selection will be required so that unimportant features are removed and potential overfitting or other modeling issues like multicollinearity can be addressed. The dimensionality reduction techniques that will be explored are linear discriminant analysis, isomap embedding, and t-distributed stochastic neighbor embedding. Each of these techniques utilizes a unique method for reducing the dimensionality of the dataset, while preserving the relationships between the data points thus aiding in training a classification model. 

A variety of classification models will be evaluated to find the best performing model. The following low complexity models will be evaluated first: logistic regression and K-Nearest Neighbors. If the performance does not meet the requirements, support vector machines and random forest will be trained. Lastly, neural networks will also be evaluated for this project. A general Scikit-Learn pipeline will be used to aid in simplifying the modeling pr


VI. Experiments & Evaluation

To evaluate the performance of the interpreter, a series of experiments will be performed. First, evaluation of the performance of the models on the test data will give a general understanding of any shortcomings. NxN matrices of the accuracy, precision, and recall will be computed. ROC curves for each label will also be plotted to see the relationship between the true positive and false positive rates and the performance comparison to the random classifier. Second, a real-time video feed of sign language via webcam will give insight on specific examples of classification. Similarly, any misclassifications in the test dataset will be observed to determine any issues with the models. Lastly, confidence scores will be calculated to help the user’s understanding of how the model is evaluating each example. 

VII. Conclusion