#Steps to perfom

# Build a recommendation engine for movies
# Do necessary preprocessing if required and follow the steps given in class room activity
#upload R file in github

#Clean your environment
rm(list=ls(all=TRUE))
#install.packages('recommenderlab')
library(recommenderlab) 

MoviesData <- read.csv(file="C:\\MOOC\\Machine Learning\\RecommendationEngine\\Movies.csv", header=TRUE, sep=",")
MoviesData$ID <- NULL

sum(is.na(MoviesData))

ratings <- as(MoviesData, "realRatingMatrix")
getRatings(ratings)

#Normalize the ratings matrix

ratings_n <- normalize(ratings)
getRatings(ratings_n)

#Convert to Dataframe
(as(ratings_n, "data.frame"))

#Visualizing Ratings Matrix
image(ratings_n, main = "Normalized Ratings")

#Binarize Rating matrix

r_b <- binarize(ratings, minRating=4)
image(r_b)

getRatings(ratings)

#Create Recommender Using IBCF and UBCF

r1 <- Recommender(ratings, method = "UBCF") #input is un-normalized  data 
r1

getModel(r1)


#Split the Model into Train and Evaluation sets

e <- evaluationScheme(ratings, method="split", train=0.7, given = 3)

#create two recommenders (user-based and item-based collaborative filtering) using the training data.

getRatings(getData(e, "train"))

#UBCF
r2 <- Recommender(getData(e, "train"), "UBCF")
r2

#IBCF
r3 <- Recommender(getData(e, "train"), "IBCF")
r3

#Predictions for the known part of the test data using the two algorithms

#Predict UBCF
p1 <- predict(r2, getData(e, "known"), type="ratings")
p1
as(p1, "list")

#Predict IBCF
p2 <- predict(r3, getData(e, "known"), type="ratings")
p2

as(p2, "list")

error <- rbind(
  calcPredictionAccuracy(p1, getData(e, "unknown")),
  calcPredictionAccuracy(p2, getData(e, "unknown"))
)
rownames(error) <- c("UBCF","IBCF")
error



