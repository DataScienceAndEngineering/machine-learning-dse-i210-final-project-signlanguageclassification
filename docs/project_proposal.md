# Sign Language Interpreter Project Proposal

### **Data Science Domains:** Computer Vision, Classification Machine Learning 

### **Authors:** Allen Lau, Sumaiya Uddin, Shubham Khandale
<br>

**I. Introduction**

**II. Motivation & Problem Statement**

**III. Related Work**

Machine learning has been used in a lot of research and development in the field of sign language recognition and interpretation. The difficulties associated with sign language recognition and interpretation have been the subject of numerous studies.

One example is from researchers at the University of Washington who developed a sign language recognition system using a combination of computer vision and machine learning. The system tested a Microsoft Kinect sensor to capture the hand and body movements of the signer and then used a Hidden Markov Model(HMM) to recognize the sign. Another example is at Carnegie Mellon University where neural networks were trained using a custom-built glove with sensors to capture the hand movements of the signer and then used to recognize the sign. Lastly, an illustration of a standard-based framework was the American Communication via Gestures (ASL) acknowledgment framework created in 1998. This framework utilized a glove with sensors to catch hand developments and perceived signs in light of predefined rules. While this framework accomplished an acknowledgment exactness of 98%, it was restricted in its capacity to perceive signs performed by various clients with differing hand sizes and shapes.

Although there are countless examples of impactful sign language interpretation modes, there is still room for improvement. There is a struggle in involving communication through signing acknowledgment frameworks for various applications, for example, gesture-based communication interpretation frameworks, communication through signing learning stages, and correspondence help for the hard of hearing and deaf. In order for people who are deaf or hard of hearing and the general public to communicate effectively, these applications need to be able to recognize sign language in real-time and accurately.

Regardless of the headway made in communication through signing acknowledgment and understanding, there still exists difficulties. For example, fluctuations in marking styles, lighting conditions, foundation mess, and impediments. In the field of applied machine learning, these issues need to be addressed as well as the accuracy and robustness of sign language recognition systems need to be improved.

**IV. Dataset & Features**

**V. Methodology**

To build the Sign Language Interpreter, the following methodology will be followed: data loading/cleaning, exploratory data analysis, preprocessing (image transformations like conversion to gray images and edge detection and feature scaling), train-test dataset splitting, dimensionality reduction, feature selection, modeling, evaluation, and interpretation. This section of the project proposal will discuss important steps in the methodology in closer detail.

Preprocessing aims to improve the performance of the model and prevent issues like overfitting. Firstly, the images will be converted to grayscale, which simplifies the images reducing the computational requirements in the data science workflow. Additionally, the Sign Language Interpreter does not need to train on the color of an individual’s hand to classify the image. Next, feature scaling ensures that models that are affected by scale are not overly affected by high pixel values, thus resulting in a lower chance of overfitting. Lastly, edge detection will be used to further decrease the complexity of the images in an attempt to improve the performance of the interpreter and decrease the runtime of training. This is due to the fact that only the edges of the hand are needed to understand which letter is being signed.

Due to the high dimensionality of the image dataset, dimensionality reduction and feature selection will be required so that unimportant features are removed and potential overfitting or other modeling issues like multicollinearity can be addressed. The dimensionality reduction techniques that will be explored are linear discriminant analysis, isomap embedding, and t-distributed stochastic neighbor embedding. Each of these techniques utilizes a unique method for reducing the dimensionality of the dataset, while preserving the relationships between the data points thus aiding in training a classification model.

A variety of classification models will be evaluated to find the best performing model. The following low complexity models will be evaluated first: logistic regression and K-Nearest Neighbors. If the performance does not meet the requirements, support vector machines and random forest will be trained. Lastly, neural networks will also be evaluated for this project.  A general Scikit-Learn pipeline will be used to aid in simplifying the modeling process. Ensemble learning techniques like XGBoost will also be explored.


**VI. Experiments & Evaluation**

To evaluate the performance of the interpreter, a series of experiments will be performed. First, evaluation of the performance of the models on the test data will give a general understanding of any shortcomings. NxN matrices of the accuracy, precision, and recall will be computed. ROC curves for each label will also be plotted to see the relationship between the true positive and false positive rates and the performance comparison to the random classifier. Second, a real-time video feed of sign language via webcam will give insight on specific examples of classification. Similarly, any misclassifications in the test dataset will be observed to determine any issues with the models. Lastly, confidence scores will be calculated to help the user’s understanding of how the model is evaluating each example.

**VII. Conclusion**