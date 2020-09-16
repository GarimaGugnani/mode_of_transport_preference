#Set Directory
getwd()
setwd("D:/R files/Machine Learning-Project")
car_dataset=read.csv("Cars_edited.csv", header=TRUE)
#Understand Data
dim(car_dataset)
head(car_dataset)
tail(car_dataset)
str(car_dataset)
car_dataset$Engineer=as.factor(car_dataset$Engineer)
car_dataset$MBA=as.factor(car_dataset$MBA)
car_dataset$license=as.factor(car_dataset$license)
car_dataset$Car_as_transport=ifelse(car_dataset$Transport =="Car",1,0)
car_dataset$Car_as_transport=as.factor(car_dataset$Car_as_transport)
any(is.na(car_dataset))
sapply(car_dataset,function(x) sum(is.na(x)))
summary(car_dataset)
#1 NA in MBA variable
#omit NA
car_dataset=na.omit(car_dataset)
prop.table(table(car_dataset$Car_as_transport))

#CORRELATION
car_corr=cor(car_dataset[,c(1,6,7)])
cor(car_dataset)
car_corr
library(corrplot)
corrplot(car_corr)


#this is showing high positive relation between age and salary
#for now, not removing any variable


library(caTools)
set.seed(123)
split <- sample.split(car_dataset$Car_as_transport, SplitRatio = 0.7)
#splitting the data such that 70% of the data is Train Data and 30% of the data is Test Data

train<- subset(car_dataset, split == TRUE)
test<- subset( car_dataset, split == FALSE)

table(train$Car_as_transport)
sum(train$Car_as_transport== 1)/nrow(train)
table(test$Car_as_transport)
sum(test$Car_as_transport== 1)/nrow(test)

car_train=train[,c(1:8,10)]
car_test=test[,c(1:8,10)]
#apply SMOTE
library(DMwR)
balanced_car<- SMOTE(Car_as_transport~., car_train, perc.over = 143, k = 5, perc.under = 600)
table(balanced_car$Car_as_transport)
prop.table(table(balanced_car$Car_as_transport))


#Logistic Regression
logistic_model1<- glm(Car_as_transport~., data=balanced_car, family=binomial(link="logit"))
summary(logistic_model1)

library(car)
vif(logistic_model1)

logistic_model2<-glm(Car_as_transport ~ . -Age -Work.Exp, 
                     data =balanced_car, 
                     family = binomial(link="logit"))

summary(logistic_model2)
vif(logistic_model2)


# Odds Ratio
exp(coef(logistic_model2))

# Probability
exp(coef(logistic_model2))/(1+exp(coef(logistic_model2)))


log.pred<-predict(logistic_model2, balanced_car[,1:8], type="response")


y_pred_num = ifelse(log.pred>0.30,1,0)
y_pred = factor(y_pred_num, levels=c(0,1))
y_actual = balanced_car$Car_as_transport
confusionMatrix(y_pred,y_actual,positive="1")

library(caret)
library(ROCR)
library(ineq)
library(InformationValue)


predobjtrain = prediction (log.pred, balanced_car$Car_as_transport)
perftrain = performance(predobjtrain, "tpr", "fpr")
plot(perftrain)#ROC curve

auc = performance(predobjtrain, "auc")
auc = as.numeric(auc@y.values)
auc

KStrain=max(perftrain@y.values[[1]]-perftrain@x.values[[1]])
KStrain


Ginitrain=ineq(log.pred, "gini")
Ginitrain

pred= predict(logistic_model2, newdata=car_test, type="response")
y_pred_num = ifelse(pred>0.30,1,0)
y_pred = factor(y_pred_num, levels=c(0,1))
y_actual = car_test$Car_as_transport
confusionMatrix(y_pred,y_actual,positive="1")


preobjtest=prediction(pred,car_test$Car_as_transport)
preftest=performance(preobjtest,"tpr","fpr")
plot(preftest)

auctest=performance(preobjtest,"auc")
auctest=as.numeric(auctest@y.values)
auctest

KStest=max(preftest@y.values[[1]]-preftest@x.values[[1]])
KStest

Ginitest=ineq(pred, "gini")
Ginitest


#KNN
library(class)
library(caret)
library(ggplot2)
scale = preProcess(balanced_car, method = "range")

train.norm.data = predict(scale, balanced_car)
test.norm.data = predict(scale, car_test)

knn_fit = train(Car_as_transport ~., data = train.norm.data, method = "knn",
                trControl = trainControl(method = "cv", number = 5),
                tuneLength = 10)

knn_fit$bestTune$k
knn_fit

knn_fit = train(Car_as_transport ~., data = train.norm.data, method = "knn",
                trControl = trainControl(method = "cv", number = 5),
                tuneLength = 5)

#knn_fit<- knn(train = balanced_car[,1:8], test = car_test[,1:8], cl= balanced_car[,9],k = 3,prob=TRUE) 
#table(car_test[,9],knn_fit)
pred_knntrain = predict(knn_fit, data = train.norm.data[,-9], type = "raw")
#confusionMatrix(pred_knntrain,train.norm.data$Car_as_transport,positive="1")
table(balanced_car[,9],pred_knntrain)


pred_knntest = predict(knn_fit, newdata = test.norm.data[,-9], type = "raw")
#confusionMatrix(pred_knntest,test.norm.data$Car_as_transport,positive="1")
table(car_test[,9],pred_knntest)


#Naive Bayes
library(e1071)

nb_car<-naiveBayes(x=balanced_car[,1:8], y=as.factor(balanced_car[,9]))
nb_car

pred_nb<-predict(nb_car,newdata = car_test[,1:8])

table(car_test[,9],pred_nb) 


#Bagging###########################################################################################################################
library(ipred)
library(rpart)

car_bagging <- bagging(Car_as_transport~.,
                          data=balanced_car,
                          control=rpart.control(maxdepth=5, minsplit=3))
car_bagging
car_test$pred.class <- predict(car_bagging,car_test)

car_test$pred.class
#car_test$pred.class<- ifelse(car_test$pred.class<0.5,0,1)
library(caret)
library(ggplot2)
confusionMatrix(data=factor(car_test$pred.class),
              reference=factor(car_test$Car_as_transport),
                positive="1")

#table(car_test$Car_as_transport,car_test$pred.class>0.5)

##########################################################################################################################################

#Boosting\
library(gbm)          # basic implementation using AdaBoost
library(xgboost)      # a faster implementation of a gbm
library(caret)

#Gradient Boosting############################################################################################################################

gbm.fit <- gbm(
  formula = Car_as_transport~ .,
  distribution = "bernoulli",
  data = balanced_car,
  n.trees = 100,
  interaction.depth = 1,
  shrinkage = 0.001,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  
gbm.fit

set.seed(123)
fitControl = trainControl(method="cv", number=5, returnResamp = "all")

model2 = train(Car_as_transport~., data=balanced_car, method="gbm",distribution="bernoulli", 
               trControl=fitControl, verbose=F, 
               tuneGrid=data.frame(.n.trees=100, .shrinkage=0.01, .interaction.depth=1, .n.minobsinnode=1))

model2


pred_class <- predict(model2, car_test)

confusionMatrix(car_test$Car_as_transport,  pred_class)


#############################################################################################################################################

#XGBoost######################################################################################################################################

car_features_train<-data.matrix(balanced_car[,1:8])
car_label_train<-data.matrix(balanced_car[,9])
car_features_test<-data.matrix(car_test[,1:8])


xgb.fit <- xgboost(
  data = car_features_train,
  label = car_label_train,
  eta = 0.001,
  max_depth = 3,
  min_child_weight = 3,
  nrounds = 10000,
  nfold = 5,
  objective = "binary:logistic",  # for regression models
  verbose = 1,               # silent,
  early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
)

#gd_features_test<-as.matrix(gd_features_test[,1:ncol(gd_features_test)-1])

car_test$xgb.pred.class <- predict(xgb.fit, car_features_test)

table(car_test$Car_as_transport,car_test$xgb.pred.class>0.5)

#or simply the total correct of the minority class
sum(car_test$Car_as_transport==1 & car_test$xgb.pred.class>=0.5)


#lets find best fit
tp_xgb<-vector()
lr <- c(0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1)
md<-c(1,3,5,7,9,15)
nr<-c(2, 50, 100, 1000, 10000)
for (i in md) {
  
  xgb.fit <- xgboost(
    data = car_features_train,
    label = car_label_train,
    eta = 0.001,
    max_depth = i,
    nrounds = 2,
    nfold = 5,
    objective = "binary:logistic",  # for regression models
    verbose = 0,               # silent,
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  
  car_test$xgb.pred.class <- predict(xgb.fit, car_features_test)
  
  tp_xgb<-cbind(tp_xgb,sum(car_test$Class==1 & car_test$xgb.pred.class>=0.5))
  
}

tp_xgb




#run code with best fit values
#now put them all into our best fit!

xgb.fit <- xgboost(
  data = car_features_train,
  label = car_label_train,
  eta = 0.001,
  max_depth = 5,
  nrounds = 50,
  nfold = 5,
  objective = "binary:logistic",  # for regression models
  verbose = 1,               # silent,
  early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
)

car_test$xgb.pred.class <- predict(xgb.fit, car_features_test)
table(car_test$Car_as_transport,car_test$xgb.pred.class>0.5)

sum(car_test$Class==1 & car_test$xgb.pred.class>=0.5)
