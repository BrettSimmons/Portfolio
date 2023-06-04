###############
## libraries ##
###############

library(readr)
library(tidyverse)
library(leaps)
library(glmnet)
library(car)
library(class)
library(caret)
library(MASS)
library(tree)

###################################################
## Data cleaning and preparation ##
###################################################

# import raw data csv as data frame 
Life_Expectancy_Data_Updated <- read_csv("Life-Expectancy-Data-Updated.csv")

#look at summary of data set
summary(Life_Expectancy_Data_Updated)
View(Life_Expectancy_Data_Updated)

#check for NA values; no NA values present
sum(is.na(Life_Expectancy_Data_Updated))

# there are 179 unique countries in our data set
length(unique(Life_Expectancy_Data_Updated$Country))

#create training set using averages of all years except 2015
life_expectancy_train<-Life_Expectancy_Data_Updated %>%
  filter(Year!=2015) %>%
  group_by(Country, Region) %>%
  summarize(across(2:19, mean)) %>%
  select(-Economy_status_Developing)


#check to make sure data frame is formatted correctly
#View(life_expectancy_train)
dim(life_expectancy_train)
summary(life_expectancy_train)
str(life_expectancy_train)

#All checks out so we can use 'life_expectancy_train' as our training set

#create test set using data from year 2015
life_expectancy_test<-Life_Expectancy_Data_Updated %>%
  filter(Year==2015) %>%
  select(-c(Year,Economy_status_Developing))



#check to make sure data frame is formatted correctly
#View(life_expectancy_test)
summary(life_expectancy_test)
dim(life_expectancy_test)
str(life_expectancy_test)

#All is good so we can use 'life_expectancy_test' as our test set


#Create new variable above_below_GDP, which indicates if a countries GDP is above or below the average GDP in the data set

mean_GDP=mean(life_expectancy_train$GDP_per_capita)

above_or_below_GDP=ifelse(life_expectancy_train$GDP_per_capita>=mean_GDP,1,0)

life_expectancy_train['above_or_below_GDP']<-above_or_below_GDP

mean_GDP=mean(life_expectancy_test$GDP_per_capita)

above_or_below_GDP=ifelse(life_expectancy_test$GDP_per_capita>=mean_GDP,1,0)

life_expectancy_test['above_or_below_GDP']<-above_or_below_GDP

###########################################
## MLR, Ridge, and Lasso ##
###########################################

### Creating scatterplots for hypothesis & to examine relationships ###
ggplot(Life_Expectancy_Data_Updated, aes(x = Infant_deaths, y = Life_expectancy)) + 
  geom_point() + 
  labs(x = "Infant deaths (per 1000 live births)", y = "Life expectancy (years)") 

ggplot(Life_Expectancy_Data_Updated, aes(x = Adult_mortality, y = Life_expectancy)) + 
  geom_point() + 
  labs(x = "Adult mortality (per 1000 population)", y = "Life expectancy (years)")

ggplot(Life_Expectancy_Data_Updated, aes(x = Schooling, y = BMI, color = Life_expectancy >= 68.68223)) + 
  geom_point() +
  scale_color_manual(values = c("orange", "darkgreen"), labels = c("Life Expectancy < 68.68", "Life Expectancy >= 68.68")) +
  labs(x = "Schooling", y = "BMI") + theme(legend.title= element_blank())

ggplot(Life_Expectancy_Data_Updated, aes(x = Hepatitis_B, y = Polio, color = Life_expectancy >= 68.68223)) + 
  geom_point() +
  scale_color_manual(values = c("orange", "darkgreen"), labels = c("Life Expectancy < 68.68", "Life Expectancy >= 68.68")) +
  labs(x = "Hepatitis B immunization (%)", y = "Polio immunization (%)") + theme(legend.position="none")

### Linear Regression ### 

mlr.model = lm(Life_expectancy~.,data=life_expectancy_train)
predicted_values = predict(mlr.model,life_expectancy_test)

MSE_test = mean((life_expectancy_test$Life_expectancy - predicted_values)^2)
MSE_test # Test MSE of 2.1146

### Ridge Regression ### 

# Lambda grid
grid = 10^seq(10,-2,length=100)

# Full data set
x = model.matrix(Life_expectancy~.,data=Life_Expectancy_Data_Updated)[,-1] 
Y = Life_Expectancy_Data_Updated$Life_expectancy

# Create training set 
x.train = model.matrix(Life_expectancy~.,data=life_expectancy_train)[,-1] 
Y.train = life_expectancy_train$Life_expectancy

# Create testing set
x.test = model.matrix(Life_expectancy~.,data=life_expectancy_test)[,-1] 
Y.test = life_expectancy_test$Life_expectancy

# Fit ridge regression model on training set
ridge.train = glmnet(x.train,Y.train,alpha=0,lambda=grid)

# Run cross-validation on training set to find optimal lambda
set.seed(23)
cv.out.ridge = cv.glmnet(x.train,Y.train,alpha = 0, lambda = grid) 
plot(cv.out.ridge)
bestlambda = cv.out.ridge$lambda.min
bestlambda

# Use optimal lambda to run prediction and find our test MSE
ridge.pred = predict(ridge.train,s=bestlambda,newx=x.test)
test.MSE = mean((ridge.pred-Y.test)^2)
test.MSE

# Run model on full dataset
final.model = glmnet(x,Y,alpha=0,lambda = bestlambda)
coef(final.model)


### Lasso Regression ###


# Fit lasso regression model on training set
lasso.train = cv.glmnet(x.train,Y.train,alpha=1,lambda=grid) 

# Run cross-validation on training set to find optimal lambda
cv.out.lasso = cv.glmnet(x.train,Y.train,alpha = 1, lambda = grid) 
plot(cv.out.lasso)
bestlambda2 = cv.out.lasso$lambda.min
bestlambda2

# Use optimal lambda for prediction and test MSE
lasso.pred = predict(lasso.train,s=bestlambda2,newx=x.test)
mean((lasso.pred-Y.test)^2)

# Run model on full dataset
final.lasso = glmnet(x,Y,alpha=1,lambda=bestlambda2)
coef(final.lasso)

### Lasso a slightly smaller test MSE than Ridge, 2.1378 vs 2.1429 ###

###########################################
## Best Subset Selection ##
###########################################

set.seed(23)

train <- life_expectancy_train[, c(-1, -2, -18)]

le <- train$Life_expectancy

test <- life_expectancy_test[, c(-1, -2, -18)]

## Best Subset Selection on training set

regfit = regsubsets(Life_expectancy~.,data=train,nbest=1,nvmax=15)
regfit.sum = summary(regfit)

n = dim(train)[1]
p = rowSums(regfit.sum$which)
adjr2 = regfit.sum$adjr2
cp = regfit.sum$cp
rss = regfit.sum$rss
AIC = n*log(rss/n) + 2*(p)
BIC = n*log(rss/n) + (p)*log(n)

cbind(p,rss,adjr2,cp,AIC,BIC)

which.min(BIC) 
which.min(AIC) 
which.min(cp)
which.max(adjr2) 

coef(regfit, which.min(BIC))

model_train = lm(Life_expectancy~Under_five_deaths + Adult_mortality + 
                   Alcohol_consumption + GDP_per_capita, data = train)

predicted_values = predict(model_train,test)
MSE_test = mean((test$Life_expectancy - predicted_values)^2)
# 2.122
MSE_test


##########################################################
## Decision Tree Classification ##
##########################################################

#------------------------------------------------------------------------------#
#Analyzing Predictor Distributions - histogram and plot

#training
plot(life_expectancy_train$GDP_per_capita)
abline(h = mean_GDP, col = "blue")

hist(life_expectancy_train$GDP_per_capita)
abline(v = mean_GDP, col = "blue")

#another visualization
plot(life_expectancy_train$Schooling,life_expectancy_train$GDP_per_capita, col = ifelse(life_expectancy_train$GDP_per_capita>=mean_GDP,'Darkgreen','orange'))


#making visualization plots based on our hypothesis
plot(life_expectancy_train$Adult_mortality,life_expectancy_train$Under_five_deaths, col = ifelse(life_expectancy_train$GDP_per_capita>=mean_GDP,'Darkgreen','Blue'))

plot(life_expectancy_train$above_or_below_GDP, col = "white")
plot(life_expectancy_train$Infant_deaths, col = ifelse(life_expectancy_train$GDP_per_capita>=mean_GDP,'Darkgreen','Blue'))
plot(life_expectancy_train$GDP_per_capita, col = ifelse(life_expectancy_train$GDP_per_capita>=mean_GDP,'Darkgreen','Blue'))
plot(life_expectancy_train$Schooling, col = ifelse(life_expectancy_train$GDP_per_capita>=mean_GDP,'Darkgreen','Blue')) 

plot(life_expectancy_train$GDP_per_capita,life_expectancy_train$Schooling ,col = ifelse(life_expectancy_train$Economy_status_Developed==1,'Darkgreen','Blue'))


#CREATING DECISION TREE CLASSIFIER --------------------------------------------#

set.seed(23)

#life = life_expectancy_train but without GDP_per_capita
#removed GDP_per_capita because it has too much correlation with above_or_below_GDP
life = life_expectancy_train[c(-1,-2,-13)]

#ensured that GDP_per_capita is a factor
life$above_or_below_GDP = as.factor(life$above_or_below_GDP)
life$Economy_status_Developed = as.factor(life$Economy_status_Developed)
str(life)

tree.GDP = tree(above_or_below_GDP~.,data=life)

summary(tree.GDP)

tree.GDP

plot(tree.GDP)
text(tree.GDP,pretty=0)

#ensemble classifier---------------------------------------------------------#

tree.gini = tree(above_or_below_GDP~., split=c("gini"), data=life)

summary(tree.GDP)
summary(tree.gini)



###########################################
## KNN Classification ##
###########################################

set.seed(23)
flds <- createFolds(life_expectancy_train$GDP_per_capita, k = 10, list = TRUE, returnTrain = FALSE)

K= c(1,3,5,7,9,11,13,15,17,19,21)

cv_error = matrix(NA, 10, 11)

## attempt using predictors identified by Amanda's single tree

#10-fold cross validation
standardized_X = scale(life_expectancy_train[,c("Under_five_deaths","Population_mln","Infant_deaths","Hepatitis_B","BMI")])

for(j in 1:11){
  k = K[j]
  for(i in 1:10){
    test_index = flds[[i]]
    testX = standardized_X[test_index,]
    trainX = standardized_X[-test_index,]
    
    trainY = life_expectancy_train$above_or_below_GDP[-test_index]
    testY = life_expectancy_train$above_or_below_GDP[test_index]
    
    knn.pred = knn(trainX,testX,trainY,k=k)
    cv_error[i,j] = mean(testY!=knn.pred)
  }
}

#optimal K = 1
optimal_k=(which.min(apply(cv_error,2,mean)))
plot(K,apply(cv_error,2,mean),ylab = "average CV error")

#KNN classification
standardized_X_test = scale(life_expectancy_test[,c("Under_five_deaths","Population_mln","Infant_deaths","Hepatitis_B","BMI")])

train_X = standardized_X
test_X = standardized_X_test
train_Y = life_expectancy_train$above_or_below_GDP
test_Y = life_expectancy_test$above_or_below_GDP

knn_pred = knn(train_X,test_X,train_Y,k=K[optimal_k])


#confusion matrix
table(knn_pred,test_Y)

#missclassification error =  0.1564246
mean(test_Y!=knn_pred)

#false positive rate = 0.1212121
false_positive = 16/(116+16)
false_positive

#attempt using predictors from ensemble decision tree

#10-fold cross validation
standardized_X = scale(life_expectancy_train[,c("Under_five_deaths","Infant_deaths","Hepatitis_B","BMI","Alcohol_consumption","Schooling","Incidents_HIV")])
standardized_X_test = scale(life_expectancy_test[,c("Under_five_deaths","Infant_deaths","Hepatitis_B","BMI","Alcohol_consumption","Schooling","Incidents_HIV")])

for(j in 1:11){
  k = K[j]
  for(i in 1:10){
    test_index = flds[[i]]
    testX = standardized_X[test_index,]
    trainX = standardized_X[-test_index,]
    
    trainY = life_expectancy_train$above_or_below_GDP[-test_index]
    testY = life_expectancy_train$above_or_below_GDP[test_index]
    
    knn.pred = knn(trainX,testX,trainY,k=k)
    cv_error[i,j] = mean(testY!=knn.pred)
  }
}

#optimal K = 19
optimal_k=(which.min(apply(cv_error,2,mean)))
plot(K,apply(cv_error,2,mean),ylab = "average CV error")

#KNN classification
train_X = standardized_X
test_X = standardized_X_test
train_Y = life_expectancy_train$above_or_below_GDP
test_Y = life_expectancy_test$above_or_below_GDP

knn_pred = knn(train_X,test_X,train_Y,k=K[optimal_k])

#confusion matrix
confusion_matrix=table(knn_pred,test_Y)
confusion_matrix
#misclassification error = 0.1564246
mean(test_Y!=knn_pred)

#false positive rate = 0.1323529
false_positive=18/(118+18)
false_positive

