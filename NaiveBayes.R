#Simple naive bayes method-------------------------------------------------
###############################################################################

# Libraries
library(naivebayes)
library(dplyr)
library(ggplot2)
library(psych)

# Data
data <- read.csv(file.choose(), header = T)
#data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", header=FALSE, sep=",", dec=".", na.strings=c("?"))
str(data)
xtabs(~admit+rank, data = data)
data$rank <- as.factor(data$rank)
data$admit <- as.factor(data$admit)

# Visualization
pairs.panels(data[-1])
data %>%
         ggplot(aes(x=admit, y=gpa, fill = admit)) +
         geom_boxplot() +
         ggtitle("Box Plot")

data %>% ggplot(aes(x=gpa, fill = admit)) +
         geom_density(alpha=0.8, color= 'black') +
         ggtitle("Density Plot")

# Data Partition way---1
#set.seed(1234)
#ind <- sample(2, nrow(data), replace = T, prob = c(0.8, 0.2))
#train <- data[ind == 1,]
#test <- data[ind == 2,]

library(caret) # data partition way-2
set.seed(123)
sample <- createDataPartition(data$admit, p = .8)[[1]]
train <- data[ sample,]
test  <- data[-sample,]

# data partition way-3
library(caTools)
sample = sample.split(data$admit, SplitRatio = .80)
train = subset(data, sample == TRUE)
test = subset(data, sample == FALSE)
print('Test train dimension')
dim(test)
dim(train)
NaiveBayesModel <- naive_bayes(V1 ~. , data = Mushrooms[training_index, ])

# Naive Bayes Model
model <- naive_bayes(admit ~ ., data = train, usekernel = F)
model

train %>%
         filter(admit == "0") %>%
         summarise(mean(gre), sd(gre))

plot(model)

# Predict
p <- predict(model, test,type='prob')
head(cbind(p, test))

# Confusion Matrix - train data
p1 <- predict(model, train)
(tab1 <- table(p1, train$admit))
1 - sum(diag(tab1)) / sum(tab1)

# Confusion Matrix - test data
p2 <- predict(model, test)
(tab2 <- table(p2, test$admit))
1 - sum(diag(tab2)) / sum(tab2)

