library(tidyverse)
library(vroom)
library(tidymodels)

#bikeTrain <- vroom("./STAT348/KaggleBikeShare/bike-sharing-demand/train.csv")
#bikeTest <- vroom("./STAT348/KaggleBikeShare/bike-sharing-demand/test.csv")

bikeTrain <- vroom("./bike-sharing-demand/train.csv")
bikeTest <- vroom("./bike-sharing-demand/test.csv")

#View(bikeTrain)
#View(bikeTest)

#Data Cleaning
#change each instance (only a single instance) of 4 to 3.
bikeTrain$weather <- ifelse(bikeTrain$weather == 4, 3, bikeTrain$weather)
#remove casual and registered columns
bikeTrain <- bikeTrain[, !(names(bikeTrain) %in% c("casual", "registered"))]


#Data Engineering

my_recipe <- recipe(count~., data=bikeTrain) %>%
  step_mutate(weather=factor(weather)) %>% #change weather to a factor
  step_mutate(season=factor(season)) %>% #change season to a factor
  step_time(datetime, features=c("hour", "minute")) %>%
  step_zv(all_predictors()) #removes zero-variance predictors

prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataS
bake(prepped_recipe, new_data=bikeTest)


#Linear Regression
my_mod <- linear_reg() %>% #Type of model
  set_engine("lm") # Engine

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = bikeTrain) # Fit the workflow

bike_pred <- predict(bike_workflow,
                            new_data=bikeTest) # Use fit to prediction

# Round all negative numbers to 0
count_rentals <- bike_pred %>%
  mutate(.pred = ifelse(.pred < 0, 0, .pred))

# extract predictions
bike_predictions <- count_rentals$.pred

# make a tibble with the test and prediction data
final <- tibble(datetime = bikeTest$datetime,
                count = bike_predictions)

# change NA to 0 and change datatype to be compatible with kaggle
final <- final %>%
  mutate(count = ifelse(is.na(count), 0, count))
final$datetime <- as.character(format(final$datetime))

# write to csv
vroom_write(final, "bike_submission.csv", delim = ",")




# Dr. Heaton Notes
# Get predictions and test set and format for kaggle
#lin_preds <- predict(bike_workflow, new_data = bikeTest) %>%
  #bind_cols(.,bikeTest) %>% # bind predictions with test data
  #select(datetime, .pred) %>% # Just keep datetime and predictions
  #rename(count=.pred) %>% # rename pred to count (for submission to Kaggle)
  #mutate(count=pmax(0,count)) %>% #pointwise max of (0,prediction)
  #mutate(datetime=as.character(format(datetime))) #needed for right format



######################
# Poisson Regression #
######################

library(poissonreg)

# create model
pois_model <- poisson_reg() %>%
  set_engine("glm") 

# create workflow with recipe and model
bike_pois_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pois_model) %>%
  fit(data = bikeTrain) # fit workflow

bike_pois_pred <- predict(bike_pois_workflow, new_data=bikeTest) %>%
  bind_cols(.,bikeTest) %>% # bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and predictions
  rename(count = .pred) %>% # rename pred to count (for submission to Kaggle)
  mutate(count = pmax(0,count)) %>% #pointwise max of (0,prediction)
  mutate(count = ifelse(is.na(count), 0, count)) %>%
  mutate(datetime=as.character(format(datetime))) #needed for right format

vroom_write(bike_pois_pred, "bike_poisson_submission.csv", delim = ',')



########################
# Penalized Regression #
########################

# transform to log
bikeTrain_log <- bikeTrain %>%
  mutate(count = log(count)) %>%
  mutate(time = as.factor(as.integer(format(bikeTrain$datetime, "%H")))) %>%
  mutate(dayofweek = as.factor(format(bikeTrain$datetime, "%A"))) 

bikeTest_new <- bikeTest %>%
  mutate(time = as.factor(as.integer(format(bikeTest$datetime, "%H")))) %>%
  mutate(dayofweek = as.factor(format(bikeTest$datetime, "%A"))) 


# penalized regression recipe
penalized_reg_recipe <- recipe(count~., data=bikeTrain_log) %>%
  step_mutate(weather=factor(weather)) %>% #change weather to a factor
  step_mutate(season=factor(season)) %>% #change season to a factor
#  step_time(datetime, features=c("hour", "minute")) %>%
  step_rm(datetime, holiday) %>% # remove datetime
  step_zv(all_predictors()) %>% #removes zero-variance predictors
  step_dummy(all_nominal_predictors()) %>% #make dummy vars
  step_normalize(all_numeric_predictors()) #make mean 0, sd=1

# set model
preg_model <- linear_reg(penalty = 0, mixture = 0) %>%
  set_engine("glmnet")

# create workflow
preg_wf <- workflow() %>%
  add_recipe(penalized_reg_recipe) %>%
  add_model(preg_model) %>%
  fit(data=bikeTrain_log)

penalized_reg_pred <- predict(preg_wf, new_data=bikeTest_new) %>%
  bind_cols(.,bikeTest) %>% # bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and predictions
  rename(count = .pred) %>% # rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>%
  mutate(count = pmax(0,count)) %>% #pointwise max of (0,prediction)
  mutate(count = ifelse(is.na(count), 0, count)) %>%
  mutate(datetime=as.character(format(datetime))) #needed for right format

vroom_write(penalized_reg_pred, "penalized_reg_submission.csv", delim = ',')



############################################################################
# Model Tuning #############################################################
############################################################################

L = 10 #levels - tells grid_regular the number of penalties and mixtures to pick
K = 10 #folds - the number of folds to split the data into

tuning_model <- linear_reg(penalty=tune(),
                           mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(penalized_reg_recipe) %>%
  add_model(tuning_model)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = L) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(bikeTrain_log, v = K, repeats=1)


## Run the CV
CV_results <- preg_wf %>%
  tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL

## Plot Results
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("rmse")

## Finalize the Workflow & fit it
final_wf <-
  preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=bikeTrain_log)

## Predict
tuning_pred <- final_wf %>%
  predict(new_data = bikeTest_new) %>%
  bind_cols(.,bikeTest) %>% # bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and predictions
  rename(count = .pred) %>% # rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>%
  mutate(count = pmax(0,count)) %>% #pointwise max of (0,prediction)
  mutate(count = ifelse(is.na(count), 0, count)) %>%
  mutate(datetime=as.character(format(datetime))) #needed for right format

vroom_write(tuning_pred, "tuning_pred.csv", delim = ',')






