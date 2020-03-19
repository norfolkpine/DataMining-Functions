
# load data
library(mlbench)
data(Sonar)
dim(Sonar)
library(gplots)
library(class)
library(mlbench)
library(caret)
library(gplots)
library (ISLR)
library(caTools)
library(naivebayes)

### Minimum
#### Libraries ####
install.packages("naivebayes")
library(naivebayes)

Mushrooms <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", header=FALSE, sep=",", dec=".", na.strings=c("?"))
summary(Mushrooms)

##### Declare Variables
no_observations <- dim(Mushrooms)[1] # No. observations (8124)
no_predictors <- dim(Mushrooms)[2] - 1 # No. predictors (22) = No. variables (23) - dependent var. (1st column)
test_index <- sample(no_observations, size=as.integer(no_observations*0.2), replace=FALSE) #20% data for test
training_index <- -test_index # Remaining 80% data observations for training

measurevar <- "V1"
preds <- colnames(Mushrooms)[2:length(Mushrooms)] #Create list containing predictors

#FUNCTION 1 : MeanError
meanError <- function(dataset, model) {
  set.seed(0)
  error <- 0
  #for (i in 2:length(Mushrooms)){
  for (i in 1:10){  
    test_index <- sample(no_observations, size=as.integer(no_observations*0.2), replace=FALSE)  # 20% data for test
    training_index <- -test_index # Remaining 80% data observations for training
    Pred_class <- predict(model, newdata = dataset[test_index, ])
    
    tab <- table(Pred_class, dataset[test_index,"V1"])
    accuracy <- sum(diag(tab))/sum(tab)
    error <- error + (1 - accuracy)
    
    #featuresA1 <- c(featuresA1, i)
  }
  (error <- error/10)
  print(error)
}

#FUNCTION 2: FIND MINIMUM COLUMN NAME
############ FIND MINIMUM / CALCULATE ERROR ###########
minimumCol <- function(predictorList) 
{
  featuresMinimum <- c()
  for (x in 1:no_predictors) 
  {
    #print(x)
    measurevar <- "V1"
    pred <- predictorList[[x]]
    
    # This creates the appropriate string:
    paste(measurevar, paste(pred, collapse=" + "), sep=" ~ ")
    
    formula <- as.formula(paste(measurevar, paste(pred, collapse=" + "), sep=" ~ "))
    #print(formula)
    
    NaiveBayesModel <- naive_bayes(formula , data = Mushrooms[training_index, ])
    errorResult <- meanError(Mushrooms, NaiveBayesModel)
    featuresMinimum <- c(featuresMinimum, errorResult)
    
    minimumIndex <- which.min(featuresMinimum)
    minimum <- preds[[minimumIndex]]
    #print(x)
    #print(minimum)
    #return(errorResult)
  }
  return(minimum)
}


selectFeature <- function(train, test, cls.train, cls.test, features) {
  ## identify a feature to be selected
  current.best.accuracy <- -Inf #nagtive infinity
  selected.i <- NULL
  for(i in 1:ncol(train)) {
    current.f <- colnames(train)[i]
    if(!current.f %in% features) {
      model <- knn(train=train[,c(features, current.f)],      test=test[,c(features, current.f)], cl=cls.train, k=3)
      test.acc <- sum(model == cls.test) / length(cls.test)
      
      if(test.acc > current.best.accuracy) {
        current.best.accuracy <- test.acc
        selected.i <- colnames(train)[i]
      }
    }
  }
  return(selected.i)
}

################### END FUNCTIONS #######################

##
library(caret)
set.seed(1)
inTrain <- createDataPartition(Sonar$Class, p = .6)[[1]]
allFeatures <- colnames(Sonar)[-61]
train <- Sonar[ inTrain,-61]
test  <- Sonar[-inTrain,-61]
cls.train <- Sonar$Class[inTrain]
cls.test <- Sonar$Class[-inTrain]


##### EDIT THIS TO BE OUR MIN FUNCTION
# use correlation to determine the first feature
cls.train.numeric <- rep(c(0, 1), c(sum(cls.train == "R"),   sum(cls.train == "M")))
cls.train.numericMush
features <- c()
current.best.cor <- 0
for(i in 1:ncol(train[,-61])) {
  if(current.best.cor < abs(cor(train[,i], cls.train.numeric))) {
    current.best.cor <- abs(cor(train[,i], cls.train.numeric))
    features <- colnames(train)[i]#  }
  }
}
##### EDIT
features <- minimumCol(allFeatures)

print(features)

# select the 2 to 10 best features using knn as a wrapper classifier
for (j in 2:10) {
  selected.i <- selectFeature(train, test, cls.train, cls.test, features)
  #print(selected.i)
  
  # add the best feature from current run
  features <- c(features, selected.i)
}
print(features)
