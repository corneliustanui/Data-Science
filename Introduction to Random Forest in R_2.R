rm(list = ls(all.names = TRUE)) # Clear loaded data
dev.off() # Reset graphical parameters to default


library(randomForest)
library(mlbench)
library(caret)

# Load Dataset
data(Sonar)
x <- Sonar[,1:60]
y <- Sonar[,61]

# Create model with default paramters
control <- trainControl(method="repeatedcv", 
                        number=10, 
                        repeats=3)
set.seed(7)
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry = mtry)
rf_default <- train(Class ~ ., 
                    data = Sonar, 
                    method = "rf", 
                    metric = "Accuracy", 
                    tuneGrid = tunegrid, 
                    trControl = control)
rf_default
plot(rf_default)

# Tuning Using Caret (Random)
control <- trainControl(method ="repeatedcv", 
                        number=10, repeats=3, 
                        search="random")
set.seed(7)
mtry <- sqrt(ncol(x))
rf_random <- train(Class ~ ., 
                   data = Sonar, 
                   method = "rf", 
                   metric = "Accuracy", 
                   tuneLength = 15, 
                   trControl = control)
rf_random
plot(rf_random)

# Tuning Using Caret (Grid)
control <- trainControl(method = "repeatedcv", 
                        number = 10, 
                        repeats = 3, 
                        search = "grid")
set.seed(7)
tunegrid <- expand.grid(.mtry = c(1:15))
rf_gridsearch <- train(Class ~ ., 
                       data = Sonar, 
                       method = "rf", 
                       metric = "Accuracy",  
                       tuneGrid = tunegrid, 
                       trControl = control)
rf_gridsearch
plot(rf_gridsearch)

# Tuning Using Algorithm (tuneRF)
set.seed(7)
bestmtry <- tuneRF(x, y, 
                   stepFactor = 1.5, 
                   improve = 1e-5, 
                   ntree=500)
bestmtry
# End
