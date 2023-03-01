# Credit_Risk_Analysis
Supervised Machine Learning and Credit Risk

All over the world, people borrow money to purchase homes or cars, start businesses, and pursue education.<br> Loans are an essential part of modern society, but loans present an opportunity and challenge for banks and other lending institutions.<br> On one hand, loans create revenue with the interest they generate. On the other hand, there's a risk that borrowers won't repay loans, and banks will lose money.<br> Banks have traditionally relied on measures like income, credit scores and collateral assets to assess lending risk, but the rise of financial technology or "Fintech" has enabled lenders to use machine learning to analyze risk.<br> Machine learning can process a large amount of data for a single decision.<br>
Fast lending, a peer-to-peer lending services company, once they use machine learning to predict credit risk, believes that this will provide a quicker and more reliable loan experience. It also believes that machine learning will lead to a more accurate identification of good candidates for loans, which will lead to lower default rates.<br>
The company asked me to assist the lead data scientist in implementing this plan.<br> We will build and evaluate several machine learning models or algorithms to predict credit risk using Python and the Scikit-learn library.<br>
Once we have designed and implemented these algorithms, our analysis will evaluate their performance and see how well our models predict data.<br>




![This is an image](https://github.com/MilosPopov007/Credit_Risk_Analysis/blob/main/Resources/FRMP_Market_small.jpg)


## Results:

Use of Resampling Models to Predict Credit Risk
  
Using the Imbalanced-learn and Scikit-learn libraries, I will evaluate three machine learning models by using resampling to determine which is better at predicting credit risk.<br> First, I will use the oversampling RandomOverSampler and SMOTE algorithms, and then the undersampling ClusterCentroids algorithm. Using these algorithms, we will :
* Resample the dataset (  credit card credit dataset from LendingClub )
* View the count of the target classes
* Train a logistic regression classifier
* Calculate the balanced accuracy score
* Generate a confusion matrix
* Generate a classification report

#### Oversampling Algorithm

Oversampling is a technique used to balance the class distribution of imbalanced datasets by increasing the number of instances in the minority class. It can be used in cases where collecting additional data is either difficult or expensive. Oversampling involves creating new synthetic instances of the minority class through various techniques such as random duplication, SMOTE (Synthetic Minority Over-sampling Technique), ADASYN (Adaptive Synthetic Sampling), and others. The goal is to create a more balanced dataset that can be used to train machine learning models that are less biased towards the majority class.

##### Naive Random Oversampling

The Naive Random Oversampling algorithm oversamples the minority class by randomly replicating samples from that class until it reaches a balance with the majority class. This method can be fast and effective when there is a clear separation between the classes, but it can also lead to overfitting and poor generalization when the classes are not well-separated.

The balanced accuracy score for the Naive Random Oversampling model is 0.6519805729802466, which indicates the model's ability to correctly classify both positive and negative cases. However, it is important to note that this score is not a complete representation of model performance and should be considered in conjunction with other evaluation metrics.

The Naive Random Oversampling algorithm resulted in a balanced accuracy score of 0.652, which is moderate. The confusion matrix shows that out of 17,205 total predictions, 11,893 of the predictions were true positives, while 5,225 were false negatives, indicating that the model has a high number of false negatives. This means that the model predicted a lower number of credit risks than it should have, resulting in more high-risk individuals being approved for credit, potentially increasing the risk of default. Therefore, while Naive Random Oversampling is a quick and easy algorithm to use, its performance could be improved.

The imbalanced classification report for Naive Random Oversampling shows that the model has a high precision rate for low-risk loans, with a value of 1.00, meaning that when the model predicts that a loan is low risk, it is almost always correct. However, the precision rate for high-risk loans is very low, with a value of only 0.01, indicating that the model is not effective at identifying high-risk loans.

The recall rate for low-risk loans is 0.69, indicating that the model correctly identified 69% of low-risk loans. However, the recall rate for high-risk loans is 0.61, indicating that the model failed to identify 39% of high-risk loans.

The f1 score for low-risk loans is 0.82, indicating that the model performs well in predicting low-risk loans. However, the f1 score for high-risk loans is very low, with a value of only 0.02, indicating that the model is not effective at identifying high-risk loans.

The geometric mean, which measures the balance between recall and specificity, is 0.65, indicating that the model is not well balanced between the two metrics.

Overall, the Naive Random Oversampling algorithm may not be the most effective method for this dataset, as it struggles to identify high-risk loans.

##### SMOTE Oversampling

SMOTE (Synthetic Minority Over-sampling Technique) is an oversampling algorithm that generates synthetic samples by randomly selecting a minority class instance and using its k-nearest neighbors to create similar, but slightly different, new instances. This technique is commonly used to address class imbalance in datasets. By generating synthetic samples, SMOTE is able to balance the class distribution while avoiding overfitting.

The balanced accuracy score for the SMOTE Oversampling algorithm is 0.6463, which is slightly lower than the Naive Random Oversampling algorithm. This means that the SMOTE algorithm was not able to improve the accuracy score as much as the Naive Random Oversampling algorithm did.
