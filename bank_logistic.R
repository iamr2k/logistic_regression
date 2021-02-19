#install.packages(c("fastDummies","knitr"))
bank <- read.csv(file.choose(),sep = ";") # Choose the bank Data set
View(bank)
sum(is.na(bank))# checking NA values in the Data 
dim(bank)
#converting catagoricalvariables to dummy

bank2 <- fastDummies::dummy_cols(bank, select_columns = 'marital')
knitr::kable(bank2)
bank2 <- fastDummies::dummy_cols(bank2, select_columns = 'education')
knitr::kable(bank2)
bank2 <- fastDummies::dummy_cols(bank2, select_columns = 'default')
knitr::kable(bank2)
bank2 <- fastDummies::dummy_cols(bank2, select_columns = 'housing')
knitr::kable(bank2)
bank2 <- fastDummies::dummy_cols(bank2, select_columns = 'loan')
knitr::kable(bank2)
bank2 <- fastDummies::dummy_cols(bank2, select_columns = 'contact')
knitr::kable(bank2)
bank2 <- fastDummies::dummy_cols(bank2, select_columns = 'month')
knitr::kable(bank2)
bank2 <- fastDummies::dummy_cols(bank2, select_columns = 'poutcome')
knitr::kable(bank2)
bank2 <- fastDummies::dummy_cols(bank2, select_columns = 'y')
knitr::kable(bank2)
View(bank2)
bank3 <- bank2[,-c(2,3,4,5,7,8,9,11,16,17,18,21,25,27,33,34,47,49,50)] #removing gender and children coloumn
View(bank3)
bank3 <- as.matrix(bank3)
bank3 <- as.data.frame(bank3)

View(bank3)
write.csv(bank3,file="bank3.csv")
data <- bank3
dim(data)
#data contains 45211 rows and 32 columns
#therefor doing PCA to reduce the processing difficulty
attach(data)
cor(data)
#use correlation matrix for getting PCA scores

pca<-princomp(data, cor = TRUE, scores = TRUE, covmat = NULL)

str(pca)
summary(pca)
loadings(pca)

plot(pca) # graph showing importance of principal components 
# Comp.1 having highest importance (highest variance)

biplot(pca)

# Showing the increase of variance with considering principal components
# Which helps in choosing number of principal components
plot(cumsum(pca$sdev*pca$sdev)*100/(sum(pca$sdev*pca$sdev)),type="b")


pca$scores[,1:5] # Top 5 PCA Scores which represents the whole data

# Creating a new df and bind the PCA data in column wise
# Considering top 5 principal component scores and binding them with bank3
bank4 <- cbind(bank3[,32],pca$scores[,1:5])
View(bank4)
# GLM function for Logistic Regression
# The output of sigmoid function lies in between 0-1

model <- glm(V1~.,data=bank4,family = "binomial")

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Confusion matrix table 
prob <- predict(model,bank4,type="response")
summary(model)
# Confusion matrix and considering the threshold value as 0.5 
confusion<-table(prob>0.5,bank4$V1)
confusion
# Model Accuracy 
Accuracy<-sum(diag(confusion)/sum(confusion))
Accuracy # 93.8


# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
yes_no <- NULL

pred_values <- ifelse(prob>=0.5,1,0)
yes_no <- ifelse(prob>=0.5,"yes","no")

 # Creating new column to store the above values
bank4[,"prob"] <- prob
bank4[,"pred_values"] <- pred_values
bank4[,"yes_no"] <- yes_no

View(bank4)

table(bank4$V1,bank4$pred_values)


# Using ROC Curve to evaluate the betterness of the logistic model
# more area under ROC curve better is the model 
library(ROCR)
rocrpred<-prediction(prob,bank4$V1)
rocrperf<-performance(rocrpred,'tpr','fpr')

str(rocrperf)

plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
# More area under the ROC Curve better is the logistic regression model obtained

## Getting cutt off or threshold value along with true positive and false positive rates in a data frame 
str(rocrperf)
rocr_cutoff <- data.frame(cut_off = rocrperf@alpha.values[[1]],fpr=rocrperf@x.values,tpr=rocrperf@y.values)
colnames(rocr_cutoff) <- c("cut_off","FPR","TPR")
View(rocr_cutoff)

library(dplyr)
rocr_cutoff$cut_off <- round(rocr_cutoff$cut_off,6)
# Sorting data frame with respect to tpr in decreasing order 
rocr_cutoff <- arrange(rocr_cutoff,desc(TPR))
View(rocr_cutoff)
