# BAYESIAN ARTIFICIAL NEURAL NETWORKS ----
rm(list = ls(all.names = TRUE))

# 1. LOADING PACKAGES ----
library(haven)             # Importing Data from Stata
library(dplyr)             # Data Manipulation
library(NeuralNetTools)    # MLE Neural Netwok
library(neuralnet)         # MLE Neural Netwok
library(nnet)              # MLE Neural Netwok
library(brnn)              # Bayesian Neural Network
library(sjstats)           # Computing correlations and associations
library(caret)             # Calculating Model Accuracy/Confusion Matrix
library(e1071)             # Calculating Confusion Matrix
library(ROCR)              # Plotting ROC and Computing AUC

#_____________________________________________________________________________
# 2. SETTING WORKING DIRECTORY AND LOADING DATASET ----
getwd()
setwd("C:\\Users\\User\\Desktop\\Robert")
dir()
data=read_dta("KEMR70FL.DTA")

#_____________________________________________________________________________
# 3. PREPARING THE DATASET ----
data1 <- data %>% select(sm811e,mv012,mv024,mv025,mv106,mv190,sm811a,mv463a)

# GENERATING VARIABLE FOR PROSTATE SPECIFIC ANTIGEN - PSA:
set.seed(500)
data1$PSA <- round(rnorm(n=12819,mean=0.255,sd=0.041),3) 

data2 <- with(data1, data.frame(y = as.numeric(sm811e[!is.na(sm811e)]),
                                x1 = as.numeric(mv012[!is.na(sm811e)]),
                                x2 = as.numeric(mv024[!is.na(sm811e)]),
                                x3 = as.numeric(mv025[!is.na(sm811e)]),
                                x4 = as.numeric(mv106[!is.na(sm811e)]),
                                x5 = as.numeric(mv190[!is.na(sm811e)]),
                                x6 = as.numeric(sm811a[!is.na(sm811e)]),
                                x7 = as.numeric(mv463a[!is.na(sm811e)]),
                                x8 = as.numeric(PSA[!is.na(sm811e)])))

rm(data); rm(data1)

# ENSURE THERE IS NO MISSING DATA
for (i in 1: ncol(data2)){
  print(any(is.na(data2[,i])))
}

# COLUMN 7 (VARIABLE X6) HAS MISSING DATA

which(is.na(data2$x6))

table(data2$x6) # WE REPLACE WITH 0 BECAUSE 0 IS THE MOST APPEARING

data2$x6 <- ifelse(is.na(data2$x6), 0, data2$x6)

# TRAINING SET
set.seed(3000)
Index=sample(2,nrow(data2),replace = TRUE, prob = c(0.7, 0.3))
Training_Set=data2[Index==1,]

# VALIDATION SET
Testing_Set=data2[Index==2,]

#_________________________________________________________________________
# 4.1  DEVELOPING THE BANN ----
BANN_Model <- brnn(y~.,
                   data = Training_Set,
                   neurones=2)

str(BANN_Model)
Parameters_BANN <- BANN_Model$theta
Parameters_BANN <- data.frame(Parameters_BANN) 
colnames(Parameters_BANN) <- c("Weights_BANN", "Biases_BANN")

View(Parameters_BANN)

# 4.2. VALIDATING THE BANN ----

# PREDICTING THE RESPONSE FOR BANN
BANN_Model_Test <- predict.brnn(BANN_Model, newdata = Testing_Set, type="prob")
BANN_Model_Test
mean(BANN_Model_Test)

# CONVERTING PROBABILITIES TO BINARY
table(BANN_Model_Test <- ifelse(BANN_Model_Test < mean(BANN_Model_Test), 0, 1))

# 4.3. MISCALSSIFICATION ----
BANN_MIS <- table(BANN_Model_Test, Testing_Set$y)

Names <- list(Predicted = c("Negative", "Positive"), Actual = c("Negative", "Positive"))
dimnames(BANN_MIS) <- Names

BANN_MIS

BANN_ACCURACY <- sum(diag(BANN_MIS))/sum(BANN_MIS)*100
BANN_ACCURACY

BANN_ACCURACY_1 <- confusionMatrix(BANN_MIS)
BANN_ACCURACY_1

# 4.4. ROC CURVE AND AUC ----
BANN_ACCURACY_2 <- prediction(BANN_Model_Test, Testing_Set$y)

BANN_Performance <- performance(BANN_ACCURACY_2, "acc")

plot(BANN_Performance, xlim = 0:1, ylim = 0:1); abline(a=0, b=1)

# 4.5. OBTAINING OPTIMAL CUTOFF ----
max_BANN <- which.max(slot(BANN_Performance, "y.values")[[1]])
acc_BANN <- slot(BANN_Performance, "y.values")[[1]][max_BANN] 
cutoff_BANN <- slot(BANN_Performance, "x.values")[[1]][max_BANN] 
print(c(Accuracy = acc_BANN, Cutoff = cutoff_BANN))
perf.BANN <- performance(BANN_ACCURACY_2, measure =  "auc", x.measure = "cutoff")
perf.BANN@y.values[[1]] <- round(perf.BANN@y.values[[1]], digits = 4)
tpr.fpr.BANN <- performance(BANN_ACCURACY_2, "tpr", "fpr")
plot(tpr.fpr.BANN, 
     colorize=TRUE, 
     xlab = "1-Specificity", 
     ylab = "Sensitivity", 
     main = "BANN ROC Curve")
abline(a=0, b=1)
text(0.6, 0.2, paste("AUC:", (perf.BANN@y.values)))


# 5.1. DEVELOPING THE MLE-ANN ----
MLE_Model <- neuralnet(y~., 
                       data = Training_Set,
                       hidden = 2,
                       threshold = 0.01,
                       err.fct = "sse",
                       act.fct = "logistic",
                       linear.output = FALSE,
                       likelihood = TRUE)
plot(MLE_Model,
     fontsize = 12,
     radius = 0.1125,
     dimesnion = 3,
     information.pos = 0.1,
     col.entry.synapse = "darkmagenta", 
     col.entry = "black",
     col.hidden = "black", 
     col.hidden.synapse = "blue",
     col.out = "black", 
     col.out.synapse = "darkviolet",
     col.intercept = "firebrick1")

garson(MLE_Model)
str(MLE_Model)
Parameters_MLE_ANN <- MLE_Model$weights
Parameters_MLE_ANN <- data.frame(Parameters_MLE_ANN) 
colnames(Parameters_MLE_ANN) <- c("Weights_MLE_ANN", "Biases_MLE_ANN")

BANN_MLE <- cbind(Parameters_BANN, rbind(Parameters_MLE_ANN, NA))

write.csv(BANN_MLE, file = "BANN_MLE.csv")


# 5.2. VALIDATING THE MLE_ANN ----

# PREDICTING THE RESPONSE FOR MLE ANN
MLE_Model_Test <- predict(MLE_Model, newdata = Testing_Set, type="prob")
MLE_Model_Test
mean(MLE_Model_Test)

# CONVERING PROBABILITIES TO BINARY
table(MLE_Model_Test <- ifelse(MLE_Model_Test < mean(MLE_Model_Test), 0, 1))

# 5.3. MISCALSSIFICATION ----
MLE_ANN_MIS <- table(MLE_Model_Test, Testing_Set$y) 

dimnames(MLE_ANN_MIS) <- Names

MLE_ANN_MIS

MLE_ANN_ACCURACY <- sum(diag(MLE_ANN_MIS))/sum(MLE_ANN_MIS)*100
MLE_ANN_ACCURACY

MLE_ANN_ACCURACY_1 <- confusionMatrix(MLE_ANN_MIS)
MLE_ANN_ACCURACY_1

# 5.4. ROC CURVE AND AUC ----
MLE_ANN_ACCURACY_2 <- prediction(MLE_Model_Test, Testing_Set$y)

MLE_ANN_Performance <- performance(MLE_ANN_ACCURACY_2, "acc")

plot(MLE_ANN_Performance, xlim = 0:1, ylim = 0:1); abline(a=0, b=1)

# 5.5. OBTAINING OPTIMAL CUTOFF ----
max_MLE_ANN <- which.max(slot(MLE_ANN_Performance, "y.values")[[1]])
acc_MLE_ANN <- slot(MLE_ANN_Performance, "y.values")[[1]][max_MLE_ANN] 
cutoff_MLE_ANN <- slot(MLE_ANN_Performance, "x.values")[[1]][max_MLE_ANN] 
print(c(Accuracy = acc_MLE_ANN, Cutoff = cutoff_MLE_ANN))
perf.MLE_ANN <- performance(MLE_ANN_ACCURACY_2, measure =  "auc", x.measure = "cutoff")
perf.MLE_ANN@y.values[[1]] <- round(perf.MLE_ANN@y.values[[1]], digits = 4)
tpr.fpr.MLE_ANN <- performance(MLE_ANN_ACCURACY_2, "tpr", "fpr")
plot(tpr.fpr.MLE_ANN, 
     colorize=TRUE,
     xlab = "1-Specificity",
     ylab = "Sensitivity",
     main = "MLE ANN ROC Curve")
abline(a=0, b=1)
text(0.6, 0.2, paste("AUC:", (perf.MLE_ANN@y.values)))

# 6. VISUALISATION ----

Bar_Data_BANN1 <- data.frame(as.factor(BANN_Model_Test))
names(Bar_Data_BANN1) <- "Status"

Bar_Data_BANN2 <- data.frame(as.factor(Testing_Set$y))
names(Bar_Data_BANN2) <- "Status"

Bar_Data_BANN <- rbind(Bar_Data_BANN1, Bar_Data_BANN2)

Bar_Data_BANN$Status <- factor(Bar_Data_BANN$Status, levels = c(0, 1), labels = c("Negative", "Positive"))
Bar_Data_BANN$Output <- NA

Bar_Data_BANN$Output[1:2397] <- "Predicted"
Bar_Data_BANN$Output[2398:4794] <- "Actual"

Bar_Data_MLE1 <- data.frame(MLE_Model_Test)
names(Bar_Data_MLE1) <- "Status"

Bar_Data_MLE2 <- data.frame(Testing_Set$y)
names(Bar_Data_MLE2) <- "Status"

Bar_Data_MLE <- rbind(Bar_Data_MLE1, Bar_Data_MLE2)
Bar_Data_MLE$Status <- factor(Bar_Data_MLE$Status, levels = c(0, 1), labels = c("Negative", "Positive"))
Bar_Data_MLE$Output <- NA
Bar_Data_MLE$Output[1:2397] <- "Predicted"
Bar_Data_MLE$Output[2398:4794] <- "Actual"

BANN_MLE <- rbind(Bar_Data_BANN, Bar_Data_MLE)
BANN_MLE$Model <- c(rep("BANN", 4794), rep("MLE", 4794))

ggplot(data = BANN_MLE, aes(x = Output, fill = Status)) + 
  scale_fill_manual(values = c("turquoise4", "darkgoldenrod2")) +
  geom_bar(position = "dodge") +
  facet_grid(~Model)+
  scale_y_continuous(breaks = seq(0, 3000, 250)) +
  ylab("Frequency (n)")+
  ggtitle("Bayesian-ANN vs MLE-ANN Prediction of Prostate Cancer Cases")

#___________________________________________________________________________________
# END OF ANALYSIS
#___________________________________________________________________________________



