# Credit_Risk_Analysis

## Project Overview

### Case for Machine Learning

The data in this project deals with credit payment information of a wide range of customers. Inherently, measuring credit risk outcomes as being clean or fraudulent has the symptom of class imbalance in the data. Meaning there is a greater likelihood of transactions being clean as opposed to being fraudulent.

This report reviews the implementation and analysis of multiple machine learning techniques to train, test and measure the accuracy of outcomes pertaining to the prediction models.

We assess the following machine learning techniques;

1. Oversampling - Random Over Sampler
2. Oversampling - SMOTE
3. Undersampling - Cluster Centroids
4. Mixed sampling - SMOTEENN
5. Balanced Random Forest Classifier
6. Easy Ensemble Classifier

This report will evaluate the balanced accuracy score, precision and recall scores pertaining to each machine learning model outputs related to the credit risk analysis.


## Review of machine learning techniques

### Oversampling - Random Over Sampler

The first machine learning model used was the Oversampling - Random Over Sampler model. Through this model we executed the following preprocessing and analytical procedures;

    * Preprocessed the data
    * Split data into training and testing
    * Resampled the data
    * Determined count of target classes
    * Trained the LRC (Logistic Regression Classifier)
    * Calculated the balanced accuracy score
    * Generated confusion matrix
    * Generated imbalanced classification report

The key data outputs of this machine learning model are as follows;

    1. Balanced Accuracy Score - 0.64
    2. Precision Score - (high - 0.01)/ (low - 1.00)/ (total - 0.99)
    3. Sensitivity (Recall) Score - (high - 0.61)/ (low - 0.68)/ (total - 0.68)

Based on the above outputs, we can determine the following;

    * The balanced accuracy score determines accuracy of prediction outputs 64% of the time
    * Considering precision (Accuracy of the Test), we see that for high risk the prediction strength is only 1%, whereas for low risk the prediction strength is 100%
    * Considering sensitivity (Likelihood or correct prediction), we see that for high risk the likelihood is 61%, whereas for low risk the likelihood is 68%

### Oversampling - SMOTE

The next machine learning model used was the Oversampling - SMOTE model. Through this model we executed the following preprocessing and analytical procedures;

    * Resampled the data
    * Determined count of target classes
    * Trained the LRC (Logistic Regression Classifier)
    * Calculated the balanced accuracy score
    * Generated confusion matrix
    * Generated imbalanced classification report

The key data outputs of this machine learning model are as follows;

    1. Balanced Accuracy Score - 0.63
    2. Precision Score - (high - 0.01)/ (low - 1.00)/ (total - 0.99)
    3. Sensitivity (Recall) Score - (high - 0.64)/ (low - 0.63)/ (total - 0.63)

Based on the above outputs, we can determine the following;

    * The balanced accuracy score determines accuracy of prediction outputs 63% of the time
    * Considering precision (Accuracy of the Test), we see that for high risk the prediction strength is only 1%, whereas for low risk the prediction strength is 100%
    * Considering sensitivity (Likelihood or correct prediction), we see that for high risk the likelihood is 64%, whereas for low risk the likelihood is 63%
    
### Undersampling - Cluster Centroids

The next machine learning model used was the Undersampling - Cluster Centroid model. Through this model we executed the following preprocessing and analytical procedures;

    * Resampled the data
    * Determined count of target classes
    * Trained the LRC (Logistic Regression Classifier)
    * Calculated the balanced accuracy score
    * Generated confusion matrix
    * Generated imbalanced classification report

The key data outputs of this machine learning model are as follows;

    1. Balanced Accuracy Score - 0.53
    2. Precision Score - (high - 0.01)/ (low - 1.00)/ (total - 0.99)
    3. Sensitivity (Recall) Score - (high - 0.61)/ (low - 0.45)/ (total - 0.45)

Based on the above outputs, we can determine the following;

    * The balanced accuracy score determines accuracy of prediction outputs 53% of the time
    * Considering precision (Accuracy of the Test), we see that for high risk the prediction strength is only 1%, whereas for low risk the prediction strength is 100%
    * Considering sensitivity (Likelihood or correct prediction), we see that for high risk the likelihood is 61%, whereas for low risk the likelihood is 45%


### Mixed sampling - SMOTEENN

Next we use the mixed (oversampling + undersampling) machine learning model of SMOTEENN model. Through this model we executed the following preprocessing and analytical procedures;

    * Resampled the data
    * Determined count of target classes
    * Trained the LRC (Logistic Regression Classifier)
    * Calculated the balanced accuracy score
    * Generated confusion matrix
    * Generated imbalanced classification report

The key data outputs of this machine learning model are as follows;

    1. Balanced Accuracy Score - 0.62
    2. Precision Score - (high - 0.01)/ (low - 1.00)/ (total - 0.99)
    3. Sensitivity (Recall) Score - (high - 0.71)/ (low - 0.54)/ (total - 0.54)

Based on the above outputs, we can determine the following;

    * The balanced accuracy score determines accuracy of prediction outputs 62% of the time
    * Considering precision (Accuracy of the Test), we see that for high risk the prediction strength is only 1%, whereas for low risk the prediction strength is 100%
    * Considering sensitivity (Likelihood or correct prediction), we see that for high risk the likelihood is 71%, whereas for low risk the likelihood is 54%

### Balanced Random Forest Classifier

Next we use the Balanced Random Forest Classifier machine learning model. Through this model we executed the following preprocessing and analytical procedures;

    * Resampled the data
    * Determined count of target classes
    * Trained the LRC (Logistic Regression Classifier)
    * Calculated the balanced accuracy score
    * Generated confusion matrix
    * Generated imbalanced classification report

The key data outputs of this machine learning model are as follows;

    1. Balanced Accuracy Score - 0.79
    2. Precision Score - (high - 0.04)/ (low - 1.00)/ (total - 0.99)
    3. Sensitivity (Recall) Score - (high - 0.67)/ (low - 0.91)/ (total - 0.67)

Based on the above outputs, we can determine the following;

    * The balanced accuracy score determines accuracy of prediction outputs 79% of the time
    * Considering precision (Accuracy of the Test), we see that for high risk the prediction strength is 4%, whereas for low risk the prediction strength is 100%
    * Considering sensitivity (Likelihood or correct prediction), we see that for high risk the likelihood is 67%, whereas for low risk the likelihood is 91%

### Easy Ensemble Classifier

Next we use the Easy Ensemble Classifier machine learning model. Through this model we executed the following preprocessing and analytical procedures;

    * Resampled the data
    * Determined count of target classes
    * Trained the LRC (Logistic Regression Classifier)
    * Calculated the balanced accuracy score
    * Generated confusion matrix
    * Generated imbalanced classification report

The key data outputs of this machine learning model are as follows;

    1. Balanced Accuracy Score - 0.92
    2. Precision Score - (high - 0.07)/ (low - 1.00)/ (total - 0.99)
    3. Sensitivity (Recall) Score - (high - 0.91)/ (low - 0.94)/ (total - 0.94)

Based on the above outputs, we can determine the following;

    * The balanced accuracy score determines accuracy of prediction outputs 92% of the time
    * Considering precision (Accuracy of the Test), we see that for high risk the prediction strength is 7%, whereas for low risk the prediction strength is 100%
    * Considering sensitivity (Likelihood or correct prediction), we see that for high risk the likelihood is 91%, whereas for low risk the likelihood is 94%

## Machine Learning Summary & Recommendations

In evaluating credit risk of customers accessing loan services, the fundamental requirement is to identify the potential of high risk customers. (Ensuring we obtain true positives) to protect the business from foreseeable losses.

Although, we need to be mindful of not labeling low risk customers incorrectly as it could potentially lose a good customer with a strong credit rating. (Minimize false negatives)

Out of the above models evaluated, we need the best model which addresses the following 2 scenarios;

1. Which model predicts true positives most accurately
2. Which model minimizes false negatives the most

### Evaluating Precision

To assess the first point, we need to look at the test which has the best prediction score. Which can be demonstrated by the following equation;

Precision = (True Positives/ True Positives + False Positives)

When considering the outcomes of the 6 machine learning techniques, all models precision scores are a 100% for low risk customers. But, given our objective, we need to ensure precision is the highest for high risk customers as we need to maximize true positives of the high risk segment.

In reviewing this, we can observe that the Easy Ensemble Classifier has the strongest output of 7%.

### Evaluating Sensitivity

To assess the second point, we need to look at the test which has the best sensitivity score. Which can be demonstrated by the following equation;

Sensitivity = (True Positives/ True Positives + False Negatives)

Assessing the above sensitivity with the objective of minimizing low risk customers from being flagged as high risk, we can see that the sensitivity score for low risk customers from the Easy Ensemble Classifier has the strongest output of 94%.

### Recommended Model

Based on the above, we can conclude that the recommended machine learning model to assess credit risk of customers to fit our business objectives precision and sensitivity tests is the Easy Ensemble Classifier model.
