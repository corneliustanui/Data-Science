# 1) LOADING DATA ----
library(dplyr)  # For glimpse()
library(randomForest)
library(caret)  # For confusionMatrix()
library(e1071)  # ML library
data_train <- read.csv("https://raw.githubusercontent.com/guru99-edu/R-Programming/master/train.csv")
glimpse(data_train)
data_test <- read.csv("https://raw.githubusercontent.com/guru99-edu/R-Programming/master/test.csv") 
glimpse(data_test)

# 2.a) DEFINE K-FOLD CROSS-VALIDATION ----

trControl <- trainControl(method = "cv", # Specify method as "cross-validation"
                          number = 10,   # Set folds to 10
                          search = "grid") # Search all combinations of hyperparameters
# 2.b) BUILD DEFAULT MODEL ----
set.seed(1234)
rf_default <- train(Survived ~ ., # Use all the other variables as predictors of survival
                    data = data_train,
                    method = "rf",       # Specify training method as "random forest"
                    metric = "Accuracy", # Chose the best model using "Accuracy" measure
                    trControl = trControl) # Use k-fold developed above to train the model


# 2.c) SEARCH THE BEST MODEl ----
set.seed(1234)
tuneGrid <- expand.grid(.mtry = c(1: 10))
rf_mtry <- train(Survived~.,
                 data = data_train,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 14,
                 ntree = 300)

# 2.d) FIND BEST VALUE OF mtry ----
best_mtry <- rf_mtry$bestTune$mtry

# 2.e) TO CHECK Accuracy
Acc <- max(rf_mtry$results$Accuracy)

# 3) FIND THE BEST maxnodes ----
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 15)) {
  set.seed(1234)
  rf_maxnode <- train(survived~.,
                      data = data_train,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 300)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)

# 4) FIND THE BEST ntrees ----
store_maxtrees <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)) {
  set.seed(5678)
  rf_maxtrees <- train(survived~.,
                       data = data_train,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 14,
                       maxnodes = 24,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)

# 5) RETRAIN THE MODEL USING OPTIMAL PARAMETERS ----
fit_rf <- train(survived~.,
                data_train,
                method = "rf",
                metric = "Accuracy",
                tuneGrid = tuneGrid,
                trControl = trControl,
                importance = TRUE,
                nodesize = 14,
                ntree = 800,     # Change ntree to optimal value, here it is 800
                maxnodes = 24)   # Change maxnodes to optimal value, here it is 24

# 6) EVALUATE THE MODEL ----
prediction <-predict(fit_rf, data_test)
confusionMatrix(prediction, data_test$survived)

# 7) VISUALISE THE MODEL  ----
varImpPlot(fit_rf)


# THE END ----






















