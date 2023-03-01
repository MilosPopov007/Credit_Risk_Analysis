# Credit_Risk_Analysis
Supervised Machine Learning and Credit Risk

All over the world, people borrow money to purchase homes or cars, start businesses, and pursue education.<br> Loans are an essential part of modern society, but loans present an opportunity and challenge for banks and other lending institutions.<br> On one hand, loans create revenue with the interest they generate. On the other hand, there's a risk that borrowers won't repay loans, and banks will lose money.<br> Banks have traditionally relied on measures like income, credit scores and collateral assets to assess lending risk, but the rise of financial technology or "Fintech" has enabled lenders to use machine learning to analyze risk.<br> Machine learning can process a large amount of data for a single decision.<br>
Fast lending, a peer-to-peer lending services company, once they use machine learning to predict credit risk, believes that this will provide a quicker and more reliable loan experience. It also believes that machine learning will lead to a more accurate identification of good candidates for loans, which will lead to lower default rates.<br>
The company asked me to assist the lead data scientist in implementing this plan.<br> We will build and evaluate several machine learning models or algorithms to predict credit risk using Python and the Scikit-learn library.<br>
Once we have designed and implemented these algorithms, our analysis will evaluate their performance and see how well our models predict data.<br>




![This is an image](https://github.com/MilosPopov007/Credit_Risk_Analysis/blob/main/Resources/FRMP_Market_small.jpg)


## Results:

### [Use of Resampling Models to Predict Credit Risk](https://github.com/MilosPopov007/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb)
  
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

Confusion matrix
True Positive (TP) = 55: The model correctly predicted that 55 loans are high-risk.
False Positive (FP) = 32: The model incorrectly predicted that 32 loans are high-risk when they are actually low-risk.
False Negative (FN) = 5812: The model incorrectly predicted that 5812 loans are low-risk when they are actually high-risk.
True Negative (TN) = 11306: The model correctly predicted that 11306 loans are low-risk.

Compared to the Naive Random Oversampling, the SMOTE Oversampling technique yielded slightly lower balanced accuracy score of 0.646. The confusion matrix shows that SMOTE correctly classified 11306 low risk loans and 55 high risk loans. However, it misclassified 32 high risk loans and 5812 low risk loans. The imbalanced classification report shows that the precision score for high risk loans is still low at 0.01, but the recall score improved to 0.63. The precision score for low risk loans is still high at 1.00, but the recall score decreased slightly to 0.66. Overall, the SMOTE Oversampling technique did not significantly improve the classification performance compared to the Naive Random Oversampling.

#### Undersampling

Undersampling is a technique that reduces the size of the majority class by randomly selecting a subset of the data points. This can help to balance the class distribution and improve model performance.

The balanced accuracy score for the undersampled data is 0.5159904274991842.

The confusion matrix for the Undersampling model shows that the model predicted 52 true positive high-risk loans and 7434 true negative low-risk loans, but misclassified 35 actual high-risk loans as low-risk and 9684 actual low-risk loans as high-risk. This means that the model has a high number of false negatives, indicating that it is not very good at identifying high-risk loans.

The balanced accuracy score for the Undersampling algorithm is the lowest among the three oversampling and undersampling methods at 0.515, which indicates that this algorithm may not perform as well as the other two. The confusion matrix shows that there are only 87 true positives, and the model misclassified 35 high-risk loans as low-risk. Additionally, there are a significant number of false negatives, with the model predicting 9,684 low-risk loans as high-risk. This result indicates that the Undersampling algorithm may not be the best choice for this particular dataset.

The imbalanced classification report shows that the precision for high-risk loans is only 0.01, indicating that when the model predicts a loan is high-risk, it is only correct 1% of the time. The recall is 0.60, meaning that the model only correctly identified 60% of the actual high-risk loans. The average f1 score is also low at 0.60, indicating that this algorithm has poor overall performance. Therefore, we should consider using another algorithm to better predict high-risk loans.

#### Combination (Over and Under) Sampling using the SMOTEENN algorithm to Predict Credit Risk

Combination Sampling is a hybrid approach that combines both Over and Under Sampling methods to address the imbalanced classification problem. The SMOTEENN algorithm uses SMOTE to oversample the minority class and Edited Nearest Neighbors (ENN) to undersample the majority class. The combination of these two techniques can improve the overall performance of the classification model by reducing both the overfitting and underfitting problems.

The balanced accuracy score for the Combination (Over and Under) Sampling using the SMOTEENN algorithm is 0.6451033596415952

Compared to the results from the previous sampling methods, the combination sampling using the SMOTEENN algorithm shows a slightly lower balanced accuracy score of 0.6451. However, the precision and recall scores for detecting high-risk loans have improved compared to the other sampling methods, with a precision score of 0.01 and recall score of 0.68. The precision score for low-risk loans is 1.00, indicating that all of the predicted low-risk loans were correct. However, the recall score for low-risk loans is lower than the previous sampling methods at 0.61, indicating that there are more false negatives. Overall, the combination sampling method has its advantages and disadvantages and may be suitable depending on the specific needs of the analysis.

### [Use of Ensemble Classifiers to Predict Credit Risk](https://github.com/MilosPopov007/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb)

Using the Imblearn.ensemble library, we will train and compare two different ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and evaluate each model. We will repeat the steps created in previous models, but this time I will be using using BalancedRandomForestClassifier and EasyEnsembleClassifier.

#### Balanced Random Forest Classifier 

The Balanced Random Forest Classifier is an ensemble learning method that uses a combination of decision trees to improve classification performance on imbalanced datasets. The algorithm creates several decision trees, each with a randomly selected subset of the features and a randomly selected subset of the training data. The final classification decision is made by taking the average of the predicted probabilities from all the trees. It is an effective method for handling class imbalance and reducing overfitting.

 The balanced accuracy score for the Balanced Random Forest Classifier is 0.7871246640962729.
 
The Balanced Random Forest Classifier performed better than all the previous models. It has a high balanced accuracy score of 0.7871, indicating that it is able to effectively classify both high-risk and low-risk loans. The recall score for high-risk loans is also the highest among all the models, although still relatively low at 0.67. The precision score for high-risk loans is low at 0.04, indicating that there are still a high number of false positives. However, the F1 score for high-risk loans has improved compared to the previous models.

#### Easy Ensemble AdaBoost Classifier

The Easy Ensemble AdaBoost Classifier is an ensemble method that combines multiple weak learners to make more accurate predictions. In this algorithm, the weak learners are decision trees that are sequentially trained to correct the mistakes made by the previous tree. This approach creates a strong classifier that can better handle imbalanced datasets. The Easy Ensemble method takes this a step further by undersampling the majority class and boosting the minority class, which helps to further balance the dataset and improve model performance.

The balanced accuracy score for the Easy Ensemble AdaBoost Classifier is 0.925427358175101. This indicates that the model performs well in predicting both classes, with a high degree of accuracy.

Based on the results, the Easy Ensemble AdaBoost Classifier outperformed the Balanced Random Forest Classifier in terms of the balanced accuracy score (0.925 vs. 0.787). Additionally, the Easy Ensemble AdaBoost Classifier had higher precision, recall, F1 score, and geometric mean for both high-risk and low-risk categories, indicating overall better performance. Therefore, the Easy Ensemble AdaBoost Classifier is the recommended algorithm for this imbalanced classification problem.


## Summary:

Naive Random Oversampling:

* Balanced accuracy score: 0.6480723054765726
* Precision and recall for high-risk loans were low (0.01 and 0.63, respectively)
* Precision and recall for low-risk loans were high (1.00 and 0.66, respectively)

Recommendation: While this model had a decent balanced accuracy score, the precision and recall for high-risk loans were very low, indicating that it may not be the best choice for identifying high-risk loans.

SMOTE Oversampling:

* Balanced accuracy score: 0.6545605858859447
* Precision and recall for high-risk loans were low (0.01 and 0.60, respectively)
* Precision and recall for low-risk loans were high (1.00 and 0.43, respectively)

Recommendation: Similar to the Naive Random Oversampling model, this model had a decent balanced accuracy score, but the precision and recall for high-risk loans were still very low. We do not recommend using this model for identifying high-risk loans.

Undersampling:

* Balanced accuracy score: 0.5159904274991842
* Precision and recall for high-risk loans were low (0.01 and 0.60, respectively)
* Precision and recall for low-risk loans were high (1.00 and 0.43, respectively)

Recommendation: The balanced accuracy score for this model was quite low, and the precision and recall for high-risk loans were still low. We do not recommend using this model for identifying high-risk loans.

Combination (Over and Under) Sampling using SMOTEENN:

* Balanced accuracy score: 0.6451033596415952
* Precision and recall for high-risk loans were low (0.01 and 0.68, respectively)
* Precision and recall for low-risk loans were high (1.00 and 0.61, respectively)

Recommendation: While the balanced accuracy score was better than the previous models, the precision and recall for high-risk loans were still not optimal. We do not recommend using this model for identifying high-risk loans.

Balanced Random Forest Classifier:

* Balanced accuracy score: 0.7871246640962729
* Precision and recall for high-risk loans were low (0.04 and 0.67, respectively)
* Precision and recall for low-risk loans were high (1.00 and 0.91, respectively)

Recommendation: This model had the highest balanced accuracy score of all the models, and while the precision and recall for high-risk loans were still not ideal, they were higher than the previous models. We recommend using this model for identifying high-risk loans.

Easy Ensemble AdaBoost Classifier:

* Balanced accuracy score: 0.925427358175101
* Precision and recall for high-risk loans were moderate (0.07 and 0.91, respectively)
* Precision and recall for low-risk loans were high (1.00 and 0.94, respectively)

Recommendation: This model had the highest balanced accuracy score and the highest recall for high-risk loans, indicating that it is the best model for identifying high-risk loans. We strongly recommend using this model for identifying high-risk loans.


The best-performing model in terms of balanced accuracy score and F1 score is the Easy Ensemble AdaBoost Classifier, with a balanced accuracy score of 0.925 and an F1 score of 0.97. This model also had the highest recall score for identifying high-risk loans, which is crucial in our case, as we want to minimize the number of high-risk loans that are incorrectly labeled as low-risk.

However, it is important to note that all models had a relatively low precision score for identifying high-risk loans, which means that there is still a risk of incorrectly labeling low-risk loans as high-risk. This could potentially result in missed opportunities for providing loans to low-risk applicants.

Additionally, we should also pay attention to the limitations of the data used to train the models. For example, there might be imbalances or biases in the data that could affect the model's performance in real-world scenarios. It is important to regularly monitor and retrain the models with updated data to ensure that they continue to perform well.

In conclusion, the Easy Ensemble AdaBoost Classifier is the recommended model to use based on its high performance in identifying high-risk loans while also maintaining a high accuracy rate. However, it is important to keep in mind the limitations of the models and the data used to train them.

