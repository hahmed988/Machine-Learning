#install.packages('forecast')
#install.packages('lubridate')
#install.packages('DataCombine')
#install.packages('imputeTS')
#install.packages('plyr')
#install.packages('dplyr')
#install.packages('TTR')
#install.packages('data.table')
#install.packages('DMwR')

#Load the Libraries
library(forecast)
library(lubridate)
library(DataCombine)
library(imputeTS)
library(plyr)
library(dplyr)
library(TTR)
library(graphics)
library(data.table)
library(DMwR)
library(TTR); 


#Read the Data
setwd("C:\\MOOC\\PHD_Scripts")
Salesdata = read.csv("Train.csv")

#Check Column Names
names(Salesdata)
nrow(Salesdata)

#Check Few lines of Data
head(Salesdata)

#Store the Target Variable 
Target <- Salesdata$Sales.In.ThousandDollars.

#Check Summary and Structure of Data
summary(Salesdata)
str(Salesdata)


##########################################################################################################
#Build Separate Time Series Model for Women Clothing

#Filter the Category for Women Clothing to build a separate Time Series 
data_women <- Salesdata[which(Salesdata$ProductCategory=='WomenClothing'), ]
head(data_women)

#Number of NA fields
sum(is.na(data_women)) #4 

#Central Imputation
data_women <- centralImputation(data_women)

#Convert the Data into TimeSeries
data_w = ts(data_women[,4],start = c(2009,1),frequency = 12)
plot(data_w, xlab='Years', ylab = 'Sales')
plot(decompose(data_w))

#Try Differencing and log of data to make the data stationary 
plot(diff(data_w),ylab='Differenced Sales(Women Clothing)')
plot(log10(data_w),ylab='Log (Sales - Women Clothing)')
plot(diff(log10(data_w)),ylab='Differenced Log (Sales - Women Clothing)')

#Plot ACF and PACF to identify potential AR and MA Model
par(mfrow = c(1,2))
acf(ts(diff(log10(data_w))),main='ACF - Sales(Women Clothing)')
pacf(ts(diff(log10(data_w))),main='PACF Sales(Women Clothing)')

#Build an Arima Model
require(forecast)
#ARIMAfit = auto.arima(log10(data_w), approximation=FALSE,trace=FALSE)
ARIMAfit = auto.arima((data_w), approximation=FALSE,trace=FALSE)
par(mfrow = c(1,1))

summary(ARIMAfit)

par(mfrow = c(1,1))
pred = predict(ARIMAfit, n.ahead = 12)
pred
plot(forecast(ARIMAfit))

#Plot ACF and PACF for residuals of ARIMA model to ensure no more information is left for extraction
par(mfrow=c(1,2))
acf(ts(ARIMAfit$residuals),main='ACF Residual')
pacf(ts(ARIMAfit$residuals),main='PACF Residual')

#Submission Code
Year = 2016
Month = 1:12
ProductCategory = 'WomenClothing'
solution1 <- data.frame('Year' = Year, 'Month' = Month , 'ProductCategory' = ProductCategory , 'target' = pred$pred)
#Write it to file
write.csv(solution1, 'SubmissionTimeSeries\\Submission1.csv', row.names = F, append=TRUE)


###########################################################################################
#Build Separate time Series for MenClothing

#Filter out Men Clothing Data 
data_men <- Salesdata[which(Salesdata$ProductCategory=='MenClothing'), ]
head(data_men)
sum(is.na(data_men)) #4

data_men <- centralImputation(data_men)
sum(is.na(data_men))

#Build Time Series
data_m = ts(data_men[,4],start = c(2009,1),frequency = 12)
plot(data_m, xlab='Years', ylab = 'Sales - Men Clothing')

#Check the 
plot(diff(data_m),ylab='Sales')
plot(log10(data_m),ylab='Log (Sales)')
plot(diff(log10(data_m)),ylab='Differenced Log (Sales)')

#Plot ACF and PACF to identify potential AR and MA Model
par(mfrow = c(1,2))
acf(ts(diff(log10(data_m))),main='ACF  Sales - Men Clothing')
pacf(ts(diff(log10(data_m))),main='PACF Sales - Men Clothing')

#ARIMAfit = auto.arima(log10(data_w), approximation=FALSE,trace=FALSE)
ARIMAfit_Men = auto.arima((data_m), approximation=FALSE,trace=FALSE, allowdrift =FALSE)
summary(ARIMAfit_Men)

par(mfrow = c(1,1))
pred_men = predict(ARIMAfit_Men, n.ahead = 12)
pred_men
#plot(data_m,type='l',xlim=c(2008,2018),ylim=c(300,1200),xlab = 'Year',ylab = 'Sales')
#lines(10^(pred_men$pred),col='blue')
#lines(10^(pred_men$pred+2*pred_men$se),col='orange')
#lines(10^(pred_men$pred-2*pred_men$se),col='orange')

plot(forecast(ARIMAfit_Men))

#Residual Plot
par(mfrow=c(1,2))
acf(ts(ARIMAfit_Men$residuals),main='ACF Residual - Men Clothing')
pacf(ts(ARIMAfit_Men$residuals),main='PACF Residual - Men Clothing')

#Submission Code
Year = 2016
Month = 1:12
ProductCategory = 'MenClothing'
solution2 <- data.frame('Year' = Year, 'Month' = Month , 'ProductCategory' = ProductCategory , 'target' = pred_men$pred)
#Write it to file
write.csv(solution2, 'SubmissionTimeSeries\\Submission2.csv', row.names = F, append=TRUE)

###########################################################################################
#Build Separate time Series for OtherClothing

install.packages('tsoutliers')
library("tsoutliers")

data_others <- Salesdata[which(Salesdata$ProductCategory=='OtherClothing'), ]
head(data_others)
sum(is.na(data_others)) #5

#Impute Missing Data 
data_others <- centralImputation(data_others)
str(data_others)
sum(is.na(data_others))

data_o = ts(data_others[,4],start = c(2009,1),frequency = 12)
plot(data_o, xlab='Years', ylab = 'Sales - Other Clothing')
plot(decompose(data_o))

plot(diff(data_o),ylab='Sales - Other Clothing')
plot(log10(data_o),ylab='Log (Sales - Other Clothing )')
plot(diff(log10(data_o)),ylab='Differenced Log (Sales - Other Clothing )')

#
par(mfrow = c(1,2))
acf(ts(((data_o))),main='ACF  Sales - Other Clothing')
pacf(ts(((data_o))),main='PACF Sales - Other Clothing')

#ARIMAfit = auto.arima(log10(data_w), approximation=FALSE,trace=FALSE)
ARIMAfit_Others = auto.arima((data_o), approximation=FALSE,trace=FALSE, allowdrift =FALSE)
summary(ARIMAfit_Others)

par(mfrow = c(1,1))
pred_o = predict(ARIMAfit_Others, n.ahead = 12)
pred_o
plot(data_o,type='l',xlim=c(2009,2017),ylim=c(1,2000),xlab = 'Year',ylab = 'Sales')
lines(10^(pred_o$pred),col='blue')
lines(10^(pred_o$pred+2*pred_o$se),col='orange')
lines(10^(pred_o$pred-2*pred_o$se),col='orange')

plot(forecast(ARIMAfit_Others))

par(mfrow=c(1,2))
acf(ts(ARIMAfit_Others$residuals),main='ACF Residual')
pacf(ts(ARIMAfit_Others$residuals),main='PACF Residual')

#Submission Code
Year = 2016
Month = 1:12
ProductCategory = 'OtherClothing'
solution3 <- data.frame('Year' = Year, 'Month' = Month , 'ProductCategory' = ProductCategory , 'target' = pred_o$pred)
#Write it to file
write.csv(solution3, 'SubmissionTimeSeries\\Submission3.csv', row.names = F, append=TRUE)

########################################|End Of Script|###################################################



