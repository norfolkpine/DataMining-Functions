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
minimumCol(preds)
