This project aims to predict the quality of red wine using chemical characteristics such as density, pH, alcohol, etc. Inspired by GeeksforGeeks "Wine Quality Prediction – Machine Learning," this project serves as a learning tool to predict data using classification. This is an evolving project. 

DATASETS:
The dataset used for this project is "winequality-red," which can be found in this repository. 

The  dataset includes the following data points relevant to red wine's chemical features:
Input variables (based on physicochemical tests):
1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulphates
11 - alcohol
12 - quality (score between 0 and 10)

MACHINE LEARNING MODELS:
This project used sklearn's RandomForestClassifier() to model the data

RESULTS:
The accuracy of the results were measure with sklearn's metrics.accuracy_score.

NEXT STEPS:
- Enhance the data pre-processing procedure
- Use additional metrics to assess prediction accuracy
- Visualize/tabulate wine quality predictions
