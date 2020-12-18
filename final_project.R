# Read csv file 


############################
## Raw data was unbalanced
raw_data <- read.csv(file='/Users/marjanrezvani/Documents/Fall2020/eco_stat/final_project/data/rawdata.csv')
colnames(raw_data)[order(colnames(raw_data))]


pie(table(raw_data$isFraud),
    labels = paste(round(prop.table(table(raw_data$isFraud))*100), "%", sep = ""), 
    col = heat.colors(2), main = "Fraud vs non-Fraud")

legend("topright", legend = c("Not-Fraud", "Fraud"), 
       fill = heat.colors(2), title = "Categories", cex = .75)

## Classification results for the unbalanced data could be misleading. So,
## I first make a balanced dataset. There are various ways to make data balanced.
## such as downsample majority class or upsample minority class.
## In this exercise, I downsample majority class.
dim(raw_data[raw_data$isFraud=='True',])[1] # Fraud 
dim(raw_data[raw_data$isFraud=='False',])[1] # non Fraud

df1 = raw_data[raw_data$isFraud=='True',]
tmp = raw_data[raw_data$isFraud=='False',]
df2 = tmp[sample(nrow(raw_data[raw_data$isFraud=='False',]),15000),]

balanced_df = rbind(df1,df2)

# Create a new variable named 'correctCVV'. Its value is True if cardCVV matches enteredCVV
balanced_df$correctCVV <- with(balanced_df, ifelse(cardCVV==enteredCVV, 'True', 'False'))

#data <- read.csv(file = '/Users/marjanrezvani/Documents/Fall2020/eco_stat/final_project/processed_balanced_transaction.csv')
#head(data)
summary(balanced_df)
attach(balanced_df)


pie(table(isFraud), labels = paste(round(prop.table(table(isFraud))*100), "%", sep = ""), 
    col = heat.colors(2), main = "Fraud vs non-Fraud")

legend("topright", legend = c("Not-Fraud", "Fraud"), 
       fill = heat.colors(2), title = "Categories", cex = 0.5)

## So from the pie chart we observe that we have kind of balanced dataset.
######################################################################################
colnames(balanced_df)[order(colnames(balanced_df))]
library(ggplot2)

## Let's do some plots to examine the dataset.
## First thing I am interested in is to explore and see if there is a difference
## between the distribution of transactionamount across Fraud or non-Fraud.
## Plot below shows that  Fradualant transavctions have higher transaction amount (e.g. 
#compare the area under the curve for range of $500 to $1000 )


ggplot(data=balanced_df, aes(x = transactionAmount, fill = isFraud)) +
  geom_density(alpha = .3) #alpha used for filling the density



# Let's see if how the transactions are distributed across different merchants.
## Plot below shows that majority of the transactions are from online_retail. fastfood, and food,
## and entertainment are the other 3 merchants with high number of transactions.
(ggplot(balanced_df, aes(x=merchantCategoryCode, fill=merchantCategoryCode))
  + geom_bar()
  + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
)

#################
# Another interesting feature is 'expirationDateKeyInMatch' which shows whether the entered
# expiration date matched the one on the card.
# Plot below shows that for fradulent transactions, majority of them expiration date does not match

(ggplot(balanced_df[balanced_df$isFraud=='True',], aes(x=expirationDateKeyInMatch, fill=expirationDateKeyInMatch))
  + geom_bar()
  + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
)




########
## Similarly, we can look at cardPresent featute. It tells whether card was present during transaction
## or not. Plot below indicates that for Fradulent transactions, most of the transactioins card 
## was not present (in other words most of them was online. Actually, we can confirm this
## by looking at distribution of Fradulent transactions across different merchants)
(ggplot(balanced_df[balanced_df$isFraud=='True',], aes(x=cardPresent, fill=cardPresent))
 + geom_bar()
 + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
)

# And Finally plot below shows that, we have most of the Fradulent transactions heppening in 
## online_retails.
(ggplot(balanced_df[balanced_df$isFraud=='True',], aes(x=merchantCategoryCode, fill=merchantCategoryCode))
  + geom_bar()
  + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
)
#########################################################################################
### Models
#balanced_df <- na.omit(balanced_df)


balanced_df$merchantCategoryCode <- as.factor(balanced_df$merchantCategoryCode)
balanced_df$transactionType <- as.factor(balanced_df$transactionType)
balanced_df$correctCVV <- as.factor(balanced_df$correctCVV)
balanced_df$cardPresent <- as.factor(balanced_df$cardPresent)





## 75% of the sample size
smp_size <- floor(0.75 * nrow(balanced_df))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(balanced_df)), size = smp_size)

train <- balanced_df[train_ind, ]
test <- balanced_df[-train_ind, ]

vartokeep <- c('isFraud','transactionAmount','merchantCategoryCode','transactionType',
               'correctCVV', 'cardPresent', 'currentBalance', 'creditLimit')
train <- train[,vartokeep]
test <- test[,vartokeep]

######################              
              
model_logit <- glm(isFraud ~ transactionAmount + merchantCategoryCode + transactionType+
                     correctCVV + cardPresent + currentBalance + creditLimit,
                    family = binomial, data = train)

summary(model_logit)

logit.pred <- predict(model_logit, test)
logit.pred_label <- (logit.pred > 0.5)

logitpredtable <- table(pred = logit.pred_label, true = test$isFraud)
logitpredtable
accuracy.logit <- sum((prop.table(logitpredtable)[1,1])+(prop.table(logitpredtable)[2,2]))
print(accuracy.logit)
#############
require('randomForest')
set.seed(54321)
model_randFor <- randomForest( as.factor(isFraud) ~ transactionAmount + merchantCategoryCode +
                                 transactionType + correctCVV + 
                                 cardPresent + currentBalance + creditLimit,
                               data = train, importance=TRUE, proximity=TRUE)

print(model_randFor)
round(importance(model_randFor),2)
varImpPlot(model_randFor)


rf.pred <- predict(model_randFor, test)


RFpredtable <- table(pred = rf.pred, true = test$isFraud)
RFpredtable
accuracy.rf <- sum((prop.table(RFpredtable)[1,1])+(prop.table(RFpredtable)[2,2]))
print(accuracy.rf)


###############
require(e1071)

svm.model <- svm(as.factor(isFraud) ~ transactionAmount + merchantCategoryCode +
                   transactionType + correctCVV + 
                   cardPresent + currentBalance + creditLimit,
                   data = train, cost = 10, gamma = 0.1, probability=TRUE)

svm.pred <- predict(svm.model, test)
SVMpredtable <- table(pred = svm.pred, true = test$isFraud)
SVMpredtable
SVMproppred <- prop.table(SVMpredtable)
SVMproppred
SVMgoodpred <- sum((SVMproppred[1,1])+(SVMproppred[2,2]))
SVMgoodpred


## Logistic Regression is a linear classifier. If the decision boundary is non-linear
## othere methods would outperform. Random Forest is an ensemble method which is capable of
## handeling non-linear boundary. SVM can also be useful when we have non-linear boundary since by
## using kernel functions (such as RBF) we can handle this non-lineartity.

## the results I obtained shows that SVM outperform logistic regression and Random Forest.




#####################
# my try:

## 












