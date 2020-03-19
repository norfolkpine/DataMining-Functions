library(naivebayes)
library(foreign)
library("dbscan")
library(reshape2)
library(ggplot2)
#install.packages("plot3D")
library("plot3D")
library(plotly)

#load the data file
getwd()
setwd("C:\\Users\\nick.moellers\\OneDrive - James Cook University\\GraduateDiploma\\MA5810-DataMining\\A3")

Stamps <- read.table("Stamps_withoutdupl_09.csv", header=FALSE, sep=",", dec=".")
summary(Stamps) # 9 Predictors (V1 to V9) and class labels (V10)
PB_Predictors <- Stamps[,1:9] # 9 Predictors (V1 to V9)
PB_class <- Stamps[,10] # Class labels (V10)
PB_class <- ifelse(PB_class == 'no',0,1) # Inliers (class "no") = 0, Outliers (class "yes") = 1

normalise <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

PB_Predictors.normal <- as.data.frame(lapply(PB_Predictors, normalise))
summary(PB_Predictors.normal)

#naive_bayes(version)

#Activity 1: Principal Component Analysis
#1 Perform Principal Component Analysis (PCA) on the Stamps data in the 9-dimensional space of the numerical predictors
pcaStamps <- prcomp(PB_Predictors, scale. = TRUE) #PCA step
pcaStamps
#show the Proportion of Variance Explained (PVE) for each of the nine resulting principal components
(PVE <- (pcaStamps$sdev^2)/sum(pcaStamps$sdev^2)) #PVE step

##Plot the accumulated sum of PVE for the first m components, as a function of m, and discuss the result:  

cumPVE <- cumsum(PVE)
plotCumPVE <- qplot(c(1:9), cumPVE) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab(NULL) + 
  ggtitle("Cumulative Sum PVE Plot") +
  ylim(0,1)
#grid.arrange(PVEplot, cumPVE, ncol = 2)
plotCumPVE
#cumsum(PVE)

#(a) How many components do we need to explain 90% or more of the total variance?
##We need 6 components to explain 93% of the total variance. If we had only 5 components, it would be 87%.

#(b) How much of the total variance is explained by the first three components?
##68.89% is explained by th
e first three components.

#2 plot a 3D scatter-plot of the Stamps data as represented by the first three principal components computed in the previous item
#( x = PC1 , y = PC2 , and z = PC3 ).
## Get answer from Stuart
pcaStamps.data3D <- as.data.frame(pcaStamps$x[,1:3])
PB_class <- as.factor(PB_class)
scatter.data <- cbind.data.frame(pcaStamps.data3D, PB_class)
#Plotly Version
plotlyResult <- plot_ly(scatter.data, x = ~PC1 , y = ~PC2, z = ~PC3, color = ~PB_class)
plotlyResult
#Scatter3D version
plotResult <- scatter3D(x=pcaStamps.data3D$PC1, y=pcaStamps.data3D$PC2, z=pcaStamps.data3D$PC3, phi = 40, theta = 40)
plotResult2 <- scatter3D(x=pcaStamps.data3D$PC1, y=pcaStamps.data3D$PC2, z=pcaStamps.data3D$PC3, phi = 20, theta = 80)
plotResult3 <- scatter3D(x=pcaStamps.data3D$PC1, y=pcaStamps.data3D$PC2, z=pcaStamps.data3D$PC3, phi = 0, theta = 10)


#do the outliers (forged stamps) look easy to detect in an unsupervised
#way, assuming that the 3D visualisation of the data via PCA is a reasonable representation of the data in
#full space? How about in a supervised way? Why? Justify your answers.
The outliers look easy to detect in an unsupervised way because the majority of stamps are clustered closely, those outside the cluster 
vary significantly in value, meaning anything outside of an acceptable range is an outlier; that is, the values deviate from the normal generating mechanisms within the cluster.

In a supervised way, it would be easy to identify forged stamps, as they would generally be very similar to those marked as forged in the training datasets


#Activity 2: Unsupervised outlier detection
#perform unsupervised outlier detection on the Stamps data in the 9-dimensional space of the numerical predictors ( PB_Predictors ), 
#using KNN Outlier with different values of the parameter (at least the following three: k =5,25,100 ).
#For each k, produce the same 3D PCA visualisation of the data as in Activity 1 (PCA), 
# - use resulting KNN Outlier Scores as a continuous, diverging colour scale. 
# -- for each k, produce asecond plot 
# --- RED: top-31 outliers according to the KNN Outlier Scores
# --- BLACK: other points
# Do these plots give you any insights on the values of that look more or less
# appropriate from an unsupervised perspective (ignoring the class labels)? Justify your answer.

library(dbscan)
k <- 5 # KNN parameter
KNN_Outlier <- kNNdist(x=PB_Predictors, k = k)[,k] # KNN distance (outlier score) computation
help("kNNdist")
#The following code sorts the observations according to their KNN outlier scores and displays the top 20 outliers along with their scores:
top_n <- 31 # No. of top outliers to be displayed
rank_KNN_Outlier <- order(x=KNN_Outlier, decreasing = TRUE) # Sorting (descending)
KNN_Result <- data.frame(ID = rank_KNN_Outlier, score = KNN_Outlier[rank_KNN_Outlier])
head(KNN_Result, top_n)
KNN_Result


# create a loop
kay <- c(5,25,100)
n <- length(kay)
for (var in 1:3){
  print(kay[var])
  KNN_Outlier <- kNNdist(x=PB_Predictors, k = kay)[,kay]
  top_n <- 31 # No. of top outliers to be displayed
  rank_KNN_Outlier <- order(x=KNN_Outlier, decreasing = TRUE) # Sorting (descending)
  KNN_Result <- data.frame(ID = rank_KNN_Outlier, score = KNN_Outlier[rank_KNN_Outlier])
  print(head(KNN_Result, 3))
  #KNN_Result
  plot(KNN_Result)
}


You have to use the knn.cv function; then use this weird function which is called rocplot or something like that 
- it's not in a package, it's one you have to define yourself. 
Basically just copy and paste that from the course notes, 
then use the output from the knn.cv function as input for the rocplot function 
which will give you the AUC for each k. I did NOT plot the rocplots, because they didn't ask for it. I gave the AUC for each k (which is what I'm pretty sure they're asking for).


#Activity 3
In this third activity you are asked to:
  1. Perform supervised classification of the Stamps data, using a KNN classifier with the same values of
as used in Activity 2 (unsupervised outlier detection). For each classifier (that is, each value of ),
compute the Area Under the Curve ROC (AUC-ROC) in a Leave-One-Out Cross-Validation (LOOCV)
scheme.

library(class)
library(ROCR)

rocplot <- function(pred, truth){
  predobj <- prediction(pred, truth)
  ROC     <- performance(predobj, "tpr", "fpr")
  plot(ROC)   # Plot the ROC Curve
  auc     <- performance(predobj, measure = "auc")
  auc     <- auc@y.values[[1]]
  return(auc) # Return the Area Under the Curve ROC
}


kay <- c(5,25,100)
n <- length(kay)
for (var in 1:3){
  Pred_class <- knn.cv(train=PB_Predictors, cl=PB_class, k=kay, prob = FALSE)
  Pred_prob <- attr(Pred_class, "prob")
  Pred_prob <- ifelse(Pred_class=='+', Pred_prob, 1 - Pred_prob) # Make sure probabilities are for class "+"
  AUC[var] <- rocplot(pred=Pred_prob, truth=PB_class)
#editing






AUC[k] <- rocplot(pred=Pred_prob, truth=class_labe
                  
2. Compare the resulting (supervised) KNN classification performance for each value of , against the
classification performance obtained in an unsupervised way by the KNN Outlier method with the same
value of . Notice that, if we rescale the KNN Outlier Scores (obtained in Activity 2 (unsupervised outlier
                                                                                      detection)) into the interval, these scores can be interpreted as outlier probabilities, which can
then be compared with the class labels (ground truth) in PB_class to compute an AUC-ROC value. This
way, for each value of , the AUC-ROC of the supervised KNN classifier can be compared with the
m
m
k k = 5, 25, 100 k
k
k
k
k
k
k
[0, 1]
k
AUC-ROC of KNN Outlier as an unsupervised classifier. Compare the performances of the supervised
versus unsupervised classifiers and discuss the results. For example, recalling that the supervised
method makes use of the class labels, whereas the unsupervised method doesn't, what can you
conclude considering there are applications where class labels are not available?