# Handling High-Dimensional Methylation Data for Parkinson's Disease Prediction

This project is to comparatively analyse different approaches to handle high-dimensional methylation data for Parkinson's disease prediction. 

The repository contains the following subdirectories and files for different parts of the project:
- preprocessing: data transformation to obtain an appriopriate format for training and testing
- training: training the models using the PPMI training data
- testing: testing the trained models using the reserved testing datasets
- clustering: feature clustering using K-means clustering applied to the selected pipeline
- evaluation: comparative evaluation of pipelines using both PPMI and PROPAG-AGEING datasets
- plotting: plotting the evaluation results for further analyses
- Documentation: recording the hyperparameter tunining results

The implementations of all models except the Multi-Layer Perceptron(MLP) in the project were done in Python version 3.6 using the Scikit-learn libraries version 0.22.1.

The MLP was implemented using Keras libraries version 2.3.1 witn tensorflow version 2.0.0
