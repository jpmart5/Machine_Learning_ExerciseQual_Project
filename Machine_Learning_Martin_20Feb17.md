---
title: "Machine Learning"
author: "Martin"
date: "February 18, 2017"
output: html_document
---

Practical Machine Learning: Course project

Data Preparation

First we'll load caret  and read in the training and testing data:

```{library}
library(caret)
## Loading required package: lattice
## Loading required package: ggplot2
```
```{trngload}
ptrain <- read.csv("C:/Users/Ryan/Desktop/Data_Science/8_Machine_Learning/pml-training.csv")
```
```{testingload}
ptest <- read.csv("C:/Users/Ryan/Desktop/Data_Science/8_Machine_Learning/pml-testing.csv")
```

In order to estimate the out-of-sample error I randomly split the full training data (ptrain) into a smaller training set (ptrain1) and a validation set (ptrain2):

```{setseed}
set.seed(10)
```
```{intrain}
inTrain <- createDataPartition(y=ptrain$classe, p=0.7, list=F)
```
```{ptrain1}
ptrain1 <- ptrain[inTrain, ]
```
```{ptrain2}
ptrain2 <- ptrain[-inTrain, ]
```

Now we need to reduce the number of features by removing: variables with close to zero variance, variables that are always NA, and variables that don't aren't useful for our prediction. We are deciding which ones to remove by analyzing ptrain1, and perform the identical removals on ptrain2:

```{novar}
nzv <- nearZeroVar(ptrain1)
# remove variables with nearly zero variance
```
```{nzv1}
ptrain1 <- ptrain1[, -nzv]
```
```{nzv2}
ptrain2 <- ptrain2[, -nzv]
```

```{justNA}
mostlyNA <- sapply(ptrain1, function(x) mean(is.na(x))) > 0.95
# remove variables that are almost always NA
```
```{NA1}
ptrain1 <- ptrain1[, mostlyNA==F]
```
```{NA2}
ptrain2 <- ptrain2[, mostlyNA==F]
```

```{PT1}
ptrain1 <- ptrain1[, -(1:5)]
# remove variables that don't make intuitive sense for prediction (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp), which happen to be the first five variables
```
```{PT2}
ptrain2 <- ptrain2[, -(1:5)]
```

Model Building

We'll start with a Random Forest model to see how it performs; fitting the model on ptrain1 and instructing the "train" function to use 3-fold cross-validation to select optimal tuning parameters for the model.

```{fitctrl}
fitControl <- trainControl(method="cv", number=3, verboseIter=F)
# instruct train to use 3-fold CV to select optimal tuning parameters
```
```{fitctrl1}
fit <- train(classe ~ ., data=ptrain1, method="rf", trControl=fitControl)
# fit model on ptrain1
```
```{randomforest}
fit$finalModel
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
# print final model to see tuning parameters it chose
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.23%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3904    1    0    0    1    0.000512
## B    5 2649    4    0    0    0.003386
## C    0    5 2391    0    0    0.002087
## D    0    0    8 2243    1    0.003996
## E    0    0    0    6 2519    0.002376
```

It evidently used 500 trees and try 27 variables at each split.

Model Evaluation and Selection

Now we fit the model to predict the label ("classe") in ptrain2, and show the confusion matrix to compare the predicted versus the actual labels:

```{predict1}
preds <- predict(fit, newdata=ptrain2)
# use model to predict classe in validation set (ptrain2)
```

```{predict2}
confusionMatrix(ptrain2$classe, preds)
# show confusion matrix to get estimate of out-of-sample error
##
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    3 1134    1    1    0
##          C    0    2 1024    0    0
##          D    0    0    2  962    0
##          E    0    0    0    2 1080
## 
## Overall Statistics
##                                         
##                Accuracy : 0.998         
##                  95% CI : (0.997, 0.999)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.998         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.998    0.997    0.997    1.000
## Specificity             1.000    0.999    1.000    1.000    1.000
## Pos Pred Value          1.000    0.996    0.998    0.998    0.998
## Neg Pred Value          0.999    1.000    0.999    0.999    1.000
## Prevalence              0.285    0.193    0.175    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.163    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.999    0.999    0.998    0.998    1.000
```

99.8% is the accuracy, so our predicted accuracy for the out-of-sample error is 0.2%.

The excellence of this result makes experimentation with other algorithms unnecessary so we'll use  Random Forests to predict on the test set.

Re-training the Selected Model

Before doing any prediction work, it's best to train the model on the full training set (ptrain), instead of using a model trained on a reduced training set (ptrain1), to produce the most accurate predictions. For that reason we're going to now repeat everything from above on ptrain and ptest:

```{remove0var}
nzv <- nearZeroVar(ptrain)
# remove variables with nearly zero variance
```
```{trainit1}
ptrain <- ptrain[, -nzv]
```
```{trainit2}
ptest <- ptest[, -nzv]
```

```{removeNA}
mostlyNA <- sapply(ptrain, function(x) mean(is.na(x))) > 0.95
# remove variables that are almost always NA
```
```{removeNA2}
ptrain <- ptrain[, mostlyNA==F]
```
```{removeNA3}
ptest <- ptest[, mostlyNA==F]
```

```{removebs1}
ptrain <- ptrain[, -(1:5)]
# remove variables that don't make intuitive sense for prediction (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp), which happen to be the first five variables
```
```{removebs2}
ptest <- ptest[, -(1:5)]
```

```{refit1}
fitControl <- trainControl(method="cv", number=3, verboseIter=F)
# re-fit model using full training set (ptrain)
```
```{refit2}
fit <- train(classe ~ ., data=ptrain, method="rf", trControl=fitControl)
```

Making Test Set Predictions

Using the model fit on ptrain to predict the label for the observations in ptest, we can write our predictions to individual files:

```{predict1}
preds <- predict(fit, newdata=ptest)
# predict on test set
```

```{predic2}
preds <- as.character(preds)
# convert predictions to character vector
```

```{predic3}
pml_write_files <- function(x) {
# create function to write predictions to files
```
```{predic4}
    n <- length(x)
```
```{predic5}
    for(i in 1:n) {
    ```
    ```{predic6}
        filename <- paste0("problem_id_", i, ".txt")
    ```
    ```{predic7}
        write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
    ```
    ```{predic8}
    }
    ```
    ```{predic9}
}
    ```
    
```{write}
pml_write_files(preds)
# create prediction files to submit
```


The Data's Source

The assignment is based on data taken from weight lifting exercises. It has been published at the following:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13). Stuttgart, Germany: ACM SIGCHI, 2013.
