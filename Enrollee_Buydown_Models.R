##NOTE: This script makes use of a sample "starter" script by Dokyun Lee, Assistant Professor of Business Analytics
##at Carnegie Mellon University. Modified with permission.

##set pwd
setwd("~/Desktop/Tepper/Data Mining Files/Final")

install.packages("ggplot2")
install.packages("AER")
library(AER)
library(caret)
library(kernlab)
library(pROC)	
library(ggplot2)

buydown_data            <- read.csv("DM_FinalProject_TrainingData_3.csv", stringsAsFactors = TRUE)

##make tier an ordered variable
buydown_data$tier       <- ordered(buydown_data$tier, levels = c("Employee Only", "Spouse", "Child","Spouse and Child"))

##remove the Id field
buydown_data            <- buydown_data[,2:8]

##review the data
str(buydown_data)
summary(buydown_data)

##plot buydown and richest cost
ggplot(buydown_data, aes(x = richest_cost, fill = buydown)) +
  geom_histogram(position = "dodge", binwidth = 100) + 
  scale_colour_hue() +
  ggtitle("Richest Cost and BuyDown")

##plot buydown and pay
ggplot(buydown_data, aes(x = pay, fill = buydown)) +
  geom_histogram(position = "dodge", binwidth = 10000) + 
  xlim(0, 200000) +
  scale_colour_hue() +
  ggtitle("Pay and BuyDown")

##plot buydown and tier
ggplot(buydown_data, aes(x = tier, fill = buydown)) +
  geom_histogram(position = "dodge", binwidth = 3, stat = "count") + 
  scale_colour_hue() +
  ggtitle("Tier and BuyDown")

####################################################
##  implement knn with 10-fold crosss validation  ##
####################################################
set.seed(100)
knn_buydown             <- train(buydown ~ ., data=buydown_data, method = "knn",
                          preProcess = c("center", "scale"), tuneGrid=expand.grid(k=1:50),
                          trControl = trainControl(method = "cv", number=10))

##ploting the accuracies of K nearest neighbor implemented for k=1 to 50
plot(knn_buydown)
knn_buydown
#generating the confusion matrix for the k nearest neighbor implemented with k having the highest accuracy
knnPredict            <- predict(knn_buydown,buydown_data)
confusionMatrix(knnPredict, buydown_data$buydown)

################################################
##                implement c5                ##
################################################
set.seed(100)
C5.0_buydown             <- train(buydown ~ ., data=buydown_data, method = "C5.0",
                           preProcess = c("center", "scale"), 
                           tuneGrid = expand.grid( .winnow = c(TRUE,FALSE), .trials=1, .model="tree" ),
                           trControl = trainControl(method = "cv", number=10))

##generating the confusion matrix for the C5.0 classification implemented in the above step
C5.0Predict             <- predict(C5.0_buydown,newdata = buydown_data)
confusionMatrix(C5.0Predict, buydown_data$buydown)

summary(C5.0_buydown)


####################################################
##        implement Support Vector Machine        ##
####################################################
set.seed(100)
svm_buydown            <- train(buydown ~ ., data=buydown_data, method = "svmLinear",
                         preProcess = c("center", "scale"),
                        trControl = trainControl(method = "cv", number=10,classProbs =  TRUE))

#generating the confusion matrix for the support vector machine implemented in the above step
svmPredict            <- predict(svm_buydown,newdata = buydown_data)
confusionMatrix(svmPredict, buydown_data$buydown)


##Installing and Loading the "ROCR" library to implement knn
install.packages("ROCR")
library(ROCR)


##Generating the roc curve for k nearest neighbor classification
##Predicting the probability of having a buydown value 'yes' from k nearest neighbor classification  
knnPredict_roc            <- predict(knn_buydown,buydown_data,type="prob")

##Prediction function combines the k nearest neighbor's estimated probability of not buydowning (Predict_roc[,2])
##and the actual buydown class (buydown_data$buydown)
pred                      <- prediction( knnPredict_roc[,2], buydown_data$buydown)

##Performance() function computes measures of performance on preciction object - pred
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")

##The preformance object obtained from the previous step is visualized using plot() function
plot(roc.perf, main="ROC curve", col="Blue", lwd=2, type="l" )

#Adding a diagonal line to the plot in the slope intercept form, a is the intercept and v is the slope
abline(a=0,b=1,lwd=2,lty=2)



##Generating the roc curve for C5.0 classification
##Predicting the probability of having a buydown value 'yes' from C5.0 classification
C5.0Predict_roc            <- predict(C5.0_buydown,newdata = buydown_data, type="prob")

##Prediction funciton combines the C5.0 classifier's estimated probability of not buydowning (Predict_roc[,2])
##and the actual buydown class (buydown_data$buydown)
pred                       <- prediction( C5.0Predict_roc[,2], buydown_data$buydown )

##Performance() function computes measures of performance on preciction object - pred
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")

##The preformance object obtained from the previous step is visualized using plot() function
plot(roc.perf, add=TRUE, main="ROC curve", col="Red", lwd=2, type="l")



##Generating the roc curve for support vector machines
##Predicting the probability of having a buydown value 'yes' from C5.0 classification
svmPredict_roc            <- predict(svm_buydown,newdata = buydown_data, type="prob")

##prediction funciton combines the support vector machines's estimated probability of not buydowning (Predict_roc[,2])
##and the actual buydown class (buydown_data$buydown)
pred                      <- prediction( svmPredict_roc[,2], buydown_data$buydown )

##Performance() function computes measures of performance on preciction object - pred
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")

##The preformance object obtained from the previous step is visualized using plot() function
plot(roc.perf, add=TRUE, main="ROC curve", col="Green", lwd=2, type="l")

#Adding the legends on the plot
legend("bottomright", legend=c("KNN", "C5.0","SVM"),
       lty=1:1, cex=0.8,col=c("blue", "red","green"))


####################################################
##                     control                    ##
####################################################
set.seed(100)
control             <- trainControl(method = "cv", number=10)
model               <- train(buydown ~ ., data=buydown_data, method="C5.0", preProcess="scale", trControl=control,
                       tuneGrid = expand.grid( .winnow = c(TRUE,FALSE), .trials=1, .model="tree" ))

##Estimatint the variable importance
importance          <- varImp(model, scale=FALSE)

##Summarize=ing the importance
print(importance)

##Plotting the importance
plot(importance)


##Selecting the top five features obtained from the previous step
svm_top5_features   <- buydown_data[,c("age", "gender", "region", "pay", "tier", "richest_cost", "buydown")]

##Implementing Support vector machine on the dataset - svm_top5_features
set.seed(100)
svm_buydown_top5     <- train(buydown ~ ., data=svm_top5_features, method = "svmLinear",
                       preProcess = c("center", "scale"),
                       trControl = trainControl(method = "cv", number=10))

#Generating the confusion matrix for the support vector machine implemented in the above step
svmPredict_top5     <- predict(svm_buydown_top5,newdata = svm_top5_features)
confusionMatrix(svmPredict_top5, buydown_data$buydown)

summary(svm_buydown_top5)

####################################################
##         test models on the holdout data        ##
####################################################

##Loading the test_data from the "DM_FinalProject_Holdout_3.csv" file, converting unknown values to 'NA' and loadings the strings as factors
new_client_data   <- read.csv("DM_FinalProject_Holdout_3.csv", stringsAsFactors = TRUE)

##create ordered factor variables by using the function ordered. 
new_client_data$tier       <- ordered(new_client_data$tier, levels = c("Employee Only", "Spouse", "Child","Spouse and Child"))

#Removing the first column which is the Cust_ID since we dont require it for our analysis
new_client_data   <- new_client_data[,2:8]

#raise the max print option to expose the full prediction output
options(max.print=10000)

##Predicting the classification of new customer based on k nearest neighbor
knnPredict             <- predict(knn_buydown, new_client_data)
knnPredict
##Predicting the classification of new customer based on C5.0
C5.0Predict            <- predict(C5.0_buydown, new_client_data)
C5.0Predict
##Predicting the classification of new customer based on svm
svmPredict             <- predict(svm_buydown, new_client_data)
svmPredict

