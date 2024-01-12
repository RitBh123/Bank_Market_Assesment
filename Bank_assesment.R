#####################Calling All the Libraries##################################
library('dplyr')
library('FSelector')
library('party')
library('rpart')
library('rpart.plot')
library('mlbench')
library('caret')
library('pROC')
library('tree')
library('C50')
library('e1071')
library('class')
library('caTools')
library('naivebayes')
library('fastDummies')
library('gmodels')

##############################Loading the CSV's##############################################################################
data <- read.csv(file.choose())
data_test <- read.csv(file.choose())
str(data)
summary(data)

######################### Modifying the CSV for use ##########################################################################

dat1 <- mutate(data,gender = factor(gender), marital_status = factor(marital_status), education = factor(education), 
               spouse_work = factor(spouse_work), residential_status = factor(residential_status), product = factor(product),
               employ_status = factor(employ_status), purchase = factor(purchase))
summary(dat1)               
str(dat1)

dat_test <- mutate(data_test,gender = factor(gender), marital_status = factor(marital_status), education = factor(education), 
               spouse_work = factor(spouse_work), residential_status = factor(residential_status), product = factor(product),
               employ_status = factor(employ_status))
summary(dat_test)               
str(dat_test)


########################## Finding info gain and other insight from the data ##################################################
info_gain <- information.gain(purchase~., data = dat1)
info_gain %>%
  arrange(desc(attr_importance))

reg_tst <- ctree(purchase ~ .,dat1)
summary(reg_tst)
plot(reg_tst)


################################ knn algorithm ###############################################################################
dum_targ <- dummy_cols(dat1,select_columns=c("education","gender","marital_status","spouse_work",
                                             "residential_status","product","employ_status","purchase"),
                       remove_first_dummy= TRUE, remove_selected_columns= TRUE)
str(dum_targ)
k_train <- dum_targ[1:25000,]
k_test <- dum_targ[25001:30000,]
k_train_label <- dum_targ[1:25000,25]
k_test_label <- dum_targ[25001:30000,25]

k_test_pred <- knn(train = k_train, test = k_test,cl = k_train_label, k=10)
CrossTable(x = k_test_label, y = k_test_pred, prop.chisq = F)


###################################### logisticreg ##############################################################################

set.seed(150000)
ind <- sample(2, nrow(dat1), replace = T, prob = c(0.8, 0.2))
train2 <- dat1[ind == 1,]
test2 <- dat1[ind == 2,]
logit <- glm(purchase~., data = train2, family = "binomial")
summary(logit)
confint.default(logit)
predict(logit, newdata = test2, type = "response")
predres <- predict(logit, newdata = test2, type = "response")
predres <- round(predres)
predres<-ifelse(predres=="0","no","yes")
predres <- as.factor(predres)
confusionMatrix(predres,test2$purchase)

test3 <- dat_test
test3 <- select(test3, -purchase)
purchase <- predict(logit, newdata = test3, type = "response")
str(purchase)
purchase <- round(purchase)
test3 <- cbind(test3,purchase)
test3$purchase<-ifelse(test3$purchase=="0","No","Yes")
test3$purchase <- as.factor(test3$purchase)
str(test3)
logit_test <- C5.0(purchase ~ .,test3)
summary(logit_test)
plot(logit_test)
write.csv(test3,"D://UCC STUFF//IS6052//Assignment//Market_pred_result_logistic.csv") 

############################################### Decision Tree CART ################################################################### 
set.seed(1234)
ind <- sample(2, nrow(dat1), replace = T, prob = c(0.8, 0.2))
train <- dat1[ind == 1,]
test <- dat1[ind == 2,]
tree <- rpart(purchase ~., data = train, cp=0.015)
rpart.rules(tree)
rpart.plot(tree)
printcp(tree)
plotcp(tree)

p <- predict(tree, test, type = 'class')
confusionMatrix(p, test$purchase)

p1 <- predict(tree, test, type = 'prob')
p1 <- p1[,2]
r <- multiclass.roc(test$purchase, p1, percent = TRUE)
roc <- r[['rocs']]
r1 <- roc[[1]]
plot.roc(r1,
         print.auc=TRUE,
         auc.polygon=TRUE,
         grid=c(0.1, 0.2),
         grid.col=c("green", "red"),
         max.auc.polygon=TRUE,
         auc.polygon.col="lightblue",
         print.thres=TRUE,
         main= 'ROC Curve')

test1 <- mutate(data_test,gender = factor(gender), marital_status = factor(marital_status), education = factor(education), 
              spouse_work = factor(spouse_work), residential_status = factor(residential_status), product = factor(product),
             employ_status = factor(employ_status))
test1$purchase <- as.factor(test1$purchase)
str(test1)
test1 <- select(test1, -purchase)
purchase <- predict(tree, test1, type = 'class')
test1 <- cbind(test1,purchase)
str(test1)
test1$purchase
reg2_tst <- C5.0(purchase ~ .,test1)
summary(reg2_tst)
plot(reg2_tst)

write.csv(test1,"D://UCC STUFF//IS6052//Assignment//Market_pred_result1.csv")


######################################## Decision Tree C5.0 ####################################################################
set.seed(1234)
ind <- sample(2, nrow(dat1), replace = T, prob = c(0.8, 0.2))
train1 <- dat1[ind == 1,]
test1 <- dat1[ind == 2,]
tree1 <- C5.0(purchase ~., data = train1, cp = 0.0025)
plot(tree1)

p5 <- predict(tree1, test1, type = 'class')
confusionMatrix(p5, test1$purchase)

p15 <- predict(tree1, test1, type = 'prob')
p15 <- p15[,2]
r5 <- multiclass.roc(test1$purchase, p15, percent = TRUE)
roc5 <- r5[['rocs']]
r15 <- roc5[[1]]
plot.roc(r15,
         print.auc=TRUE,
         auc.polygon=TRUE,
         grid=c(0.1, 0.2),
         grid.col=c("green", "red"),
         max.auc.polygon=TRUE,
         auc.polygon.col="lightblue",
         print.thres=TRUE,
         main= 'ROC Curve')

test2 <- mutate(data_test,gender = factor(gender), marital_status = factor(marital_status), education = factor(education), 
                spouse_work = factor(spouse_work), residential_status = factor(residential_status), product = factor(product),
                employ_status = factor(employ_status))
test2$purchase <- as.factor(test2$purchase)
str(test2)
test2 <- select(test2, -purchase)
purchase <- predict(tree1, test2, type = 'class')
test2 <- cbind(test2,purchase)
str(test2)
test2$purchase
reg3_tst <- C5.0(purchase ~ .,test2)
summary(reg3_tst)
plot(reg3_tst)

write.csv(test2,"D://UCC STUFF//IS6052//Assignment//Market_pred_result2.csv")

############################################### SVM ############################################################################
set.seed(1234)
ind <- sample(2, nrow(dat1), replace = T, prob = c(0.8, 0.2))
train2 <- dat1[ind == 1,]
test2 <- dat1[ind == 2,]
svmtest <- svm(purchase ~., data = train2)

summary(svmtest)
print(svmtest)
svmpred <- predict(svmtest,test2, type = 'class')
confusionMatrix(svmpred, test2$purchase)
plot(svmpred,test2$purchase)

psvm <- predict(svmtest, test2, type = 'prob')
plot(psvm)

test3 <- mutate(data_test,gender = factor(gender), marital_status = factor(marital_status), education = factor(education), 
                spouse_work = factor(spouse_work), residential_status = factor(residential_status), product = factor(product),
                employ_status = factor(employ_status))
test3$purchase <- as.factor(test3$purchase)
str(test3)
test3 <- select(test3, -purchase)
purchase <- predict(svmtest, test3, type = 'class')
test3 <- cbind(test3,purchase)
str(test3)
test3$purchase
reg4_tst <- C5.0(purchase ~ .,test3)
summary(reg4_tst)
plot(reg4_tst)

write.csv(test3,"D://UCC STUFF//IS6052//Assignment//Market_pred_result_svm.csv")

############################################naivebayes##########################################################################
set.seed(1234)
ind <- sample(2, nrow(dat1), replace = T, prob = c(0.8, 0.2))
train3 <- dat1[ind == 1,]
test3 <- dat1[ind == 2,]
naive_test <- naive_bayes(purchase ~., data = train3)

summary(naive_test)
print(naive_test)
plot(naive_test)
naivepred <- predict(naive_test,test2, type = 'class')
confusionMatrix(naivepred, test2$purchase)
plot(naivepred,test2$purchase)

pnb <- predict(naive_test, test3, type = 'prob')
plot(pnb)
test4 <- mutate(data_test,gender = factor(gender), marital_status = factor(marital_status), education = factor(education), 
                spouse_work = factor(spouse_work), residential_status = factor(residential_status), product = factor(product),
                employ_status = factor(employ_status))
test4$purchase <- as.factor(test4$purchase)
str(test4)
test4 <- select(test4, -purchase)
purchase <- predict(naive_test, test4, type = 'class')
test4 <- cbind(test4,purchase)
str(test4)
test4$purchase
reg5_tst <- C5.0(purchase ~ .,test4)
summary(reg5_tst)
plot(reg5_tst)

write.csv(test3,"D://UCC STUFF//IS6052//Assignment//Market_pred_result_nb.csv")

############################################### Decision Tree ID3 ################################################################### 
set.seed(1234)
ind <- sample(2, nrow(dat1), replace = T, prob = c(0.8, 0.2))
train <- dat1[ind == 1,]
test <- dat1[ind == 2,]
treex <- ctree(purchase ~., data = train)
print(treex)
plot(treex)

p <- predict(treex, test, type = 'response')
confusionMatrix(p, test$purchase)

test1x <- mutate(data_test,gender = factor(gender), marital_status = factor(marital_status), education = factor(education), 
                spouse_work = factor(spouse_work), residential_status = factor(residential_status), product = factor(product),
                employ_status = factor(employ_status))
test1x$purchase <- as.factor(test1x$purchase)
str(test1x)
test1x <- select(test1x, -purchase)
purchase <- predict(treex, test1x, type = 'response')
test1x <- cbind(test1x,purchase)
str(test1x)
test1x$purchase
reg2_tstx <- C5.0(purchase ~ .,test1x)
summary(reg2_tstx)
plot(reg2_tstx)

write.csv(test1x,"D://UCC STUFF//IS6052//Assignment//Market_pred_result_ID3.csv")
