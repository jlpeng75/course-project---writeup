library(caret)
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
classe <- training$classe
dim(training);dim(testing)
x <- sapply(training, is.numeric)
training <- training[, x]
dim(training)
variables <- names(training)
max <- grep("^max", variables, value = T)
min <- grep("^min", variables, value = T)
avg <- grep("^avg", variables, value = T)
var <- grep("^var", variables, value = T)
std <- grep("^stddev", variables, value = T)
amp <- grep("^amplitude", variables, value = T)
NAs <- c(max, min, avg, var, std, amp, amp)
val <- variables[! variables %in% NAs]
training <- training[, val]
training <- training[, -1]
training <- cbind(training, classe)
head(training)
modFit <- train(classe ~., method = "rf", data = training)
testing <- testing[, val]
testing <- testing[, -1]

Mod1 <-train(classe ~., method = "rf", trControl = trainControl(method = "cv", number = 10), ntree = 200, data = training)
prediction <- predict(Mod1, testing)
print(Mod1$finalModel)
prediction