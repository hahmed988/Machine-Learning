#1.predict Total revenue generated on Given Customer dataset(knn regression)
# do necessary preprocessing
# try both standardized and un-standardized


#2.Bin target variable in to two classes based on revenue generated And aplly knn classification(knn classification)


#3.upload R file in GitHUb


#Libraries

library('vegan')
library('dummy')
#Clear the workspace

rm(list=ls(all=TRUE))

#Set the working directory

setwd("C:\\MOOC\\Insofe\\Module 3 - Methods and Algo in Machine Learning\\KNN - Ensemble\\20170729-batch29-cse7305c-knn-cf-assignment-hahmed988-master (1)\\20170729-batch29-cse7305c-knn-cf-assignment-hahmed988-master")

#Read the data
Custdata = read.csv(file="CustomerData.csv", header=TRUE, sep=",")
summary(Custdata)
sum(is.na(Custdata))
str(Custdata)
Custdata$CustomerID <- NULL


#Randomly sample 70% of the records for train data
set.seed(123) 

train = sample(1:nrow(Custdata),nrow(Custdata)*0.7) 
CustData_train = Custdata[train,] 
CustData_test = Custdata[-train,] 

#Get the train and test sets excluding the target variable
Custdata_trainwithoutclass = subset(CustData_train,select=-c(TotalRevenueGenerated, FavoriteChannelOfTransaction, FavoriteGame))
Custdata_testwithoutclass = subset(CustData_test,select=-c(TotalRevenueGenerated, FavoriteChannelOfTransaction, FavoriteGame))

#"Fast Nearest Neighbours" for knn regression
install.packages("FNN") 
#to calculate error metrics for regression
install.packages("Metrics") 
library(FNN)
library(Metrics)


# Code to implement regression using knn --knn.reg

pred <- knn.reg(train = Custdata_trainwithoutclass, test = Custdata_testwithoutclass, y = CustData_train$TotalRevenueGenerated, k = 3)

# Evaluate the predictions

actual <- CustData_train$TotalRevenueGenerated
pred <- data.frame(pred$pred)

result2 <- rmse(actual = actual, predicted = pred)
result2

#Implement Knn Regression after standardizing the variables

Custdata_Stdtrain = subset(CustData_train,select=-c(TotalRevenueGenerated, FavoriteChannelOfTransaction, FavoriteGame))
Custdata_Stdtest = subset(CustData_test,select=-c(TotalRevenueGenerated, FavoriteChannelOfTransaction, FavoriteGame))

#Custdata_Stdtrain$Channel <- dummy(Custdata_Stdtrain$FavoriteChannelOfTransaction)
#Custdata_Stdtrain1 = subset(Custdata_Stdtrain,select=-c(FavoriteChannelOfTransaction)) 
#Custdata_Stdtrain = cbind(Custdata_Stdtrain1,Channel)

#Game = dummy(Custdata_Stdtrain$FavoriteGame)
#Custdata_Stdtrain2 = subset(Custdata_Stdtrain,select=-c(FavoriteGame)) 
#Custdata_Stdtrain = cbind(Custdata_Stdtrain2,Game)

Custdata_Stdtrain=decostand(Custdata_Stdtrain,"range") 
Custdata_Stdtest=decostand(Custdata_Stdtrain,"range") 

pred <- knn.reg(train = Custdata_Stdtrain, test = Custdata_Stdtest, y = CustData_train$TotalRevenueGenerated, k = 3)

# Evaluate the predictions

actual <- CustData_train$TotalRevenueGenerated
pred <- data.frame(pred$pred)

result2 <- rmse(actual = actual, predicted = pred)
result2


#Knn Classification

Custdata$TotalRevenueGenerated <- ifelse(Custdata$TotalRevenueGenerated > 500, yes = 1, no = 0)
set.seed(124) 

train = sample(1:nrow(Custdata),nrow(Custdata)*0.7) 
CustData_trainClassification = Custdata[train,] 
CustData_testClassification = Custdata[-train,] 

table(Custdata$TotalRevenueGenerated)


Custdata_trainWithoutClass = subset(CustData_trainClassification,select=-c(TotalRevenueGenerated))
Custdata_testWithoutClass = subset(CustData_testClassification,select=-c(TotalRevenueGenerated))

pred3 <- knn(train = Custdata_trainWithoutClass, test = Custdata_testWithoutClass, y = CustData_trainClassification$TotalRevenueGenerated, k = 3)


a=table(CustData_testClassification$TotalRevenueGenerated, pred3)

a
accu=sum(diag(a))/nrow(CustData_testClassification)
accu