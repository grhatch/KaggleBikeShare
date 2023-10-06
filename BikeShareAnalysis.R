library(tidyverse)
library(vroom)
library(tidymodels)
library(stacks)
library(poissonreg)


bikeTrain <- vroom("./STAT348/KaggleBikeShare/bike-sharing-demand/train.csv")
bikeTest <- vroom("./STAT348/KaggleBikeShare/bike-sharing-demand/test.csv")

#bikeTrain <- vroom("./bike-sharing-demand/train.csv")
#bikeTest <- vroom("./bike-sharing-demand/test.csv")

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



####################
# Regression Trees #
####################


regtree_mod <- decision_tree(tree_depth = tune(), #tune() = computer automatically figuring this out
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # Engine = What R function to use
  set_mode("regression")

regtree_recipe <- recipe(count~., data=bikeTrain_log) %>%
  step_mutate(weather=factor(weather)) %>% #change weather to a factor
  step_mutate(season=factor(season))  #change season to a factor
  #  step_time(datetime, features=c("hour", "minute")) %>%
  #step_rm(datetime, holiday) %>% # remove datetime
  #step_zv(all_predictors()) %>% #removes zero-variance predictors
  #step_dummy(all_nominal_predictors()) %>% #make dummy vars
  #step_normalize(all_numeric_predictors()) #make mean 0, sd=1

# set up workflow
regtree_wf <- workflow() %>%
  add_recipe(regtree_recipe) %>%
  add_model(regtree_mod)

L <- 5
## Grid of values to tune over; these should be params in the model
tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = L) ## L^2 total tuning possibilities

K <- 5
## Split data for CV
folds <- vfold_cv(bikeTrain_log, v = K, repeats=1)

## Run CV
CV_results <- regtree_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq))

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("rmse")


## Plot Results
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

## Finalize the Workflow & fit it
final_wf <-
  regtree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=bikeTrain_log)

## Predict
regtree_pred <- final_wf %>%
  predict(new_data = bikeTest_new) %>%
  bind_cols(.,bikeTest) %>% # bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and predictions
  rename(count = .pred) %>% # rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>%
  mutate(count = pmax(0,count)) %>% #pointwise max of (0,prediction)
  mutate(count = ifelse(is.na(count), 0, count)) %>%
  mutate(datetime=as.character(format(datetime))) #needed for right format

vroom_write(regtree_pred, "regtree_pred.csv", delim = ',')


#################
# Random Forest #
#################


# "Bootstrap" = sample with replacement

# transform to log
bikeTrain_log <- bikeTrain %>%
  mutate(count = log(count)) %>%
  mutate(time = as.factor(as.integer(format(bikeTrain$datetime, "%H")))) %>%
  mutate(dayofweek = as.factor(format(bikeTrain$datetime, "%A"))) %>%
  select(-casual, -registered)


bikeTest_new <- bikeTest %>%
  mutate(time = as.factor(as.integer(format(bikeTest$datetime, "%H")))) %>%
  mutate(dayofweek = as.factor(format(bikeTest$datetime, "%A"))) 

rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

rf_recipe <- recipe(count~., data=bikeTrain_log) %>%
  step_mutate(weather=factor(weather)) %>% #change weather to a factor
  step_mutate(season=factor(season)) %>%  #change season to a factor
  step_rm(weather)

# set up workflow
rf_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_mod)

L <- 5
## Grid of values to tune over; these should be params in the model
rf_tuning_grid <- grid_regular(mtry(range = c(1,10)),
                            min_n(),
                            levels = L) ## L^2 total tuning possibilities

K <- 5
## Split data for CV
rf_folds <- vfold_cv(bikeTrain_log, v = K, repeats=1)

## Run CV
rf_CV_results <- rf_wf %>%
  tune_grid(resamples=rf_folds,
            grid=rf_tuning_grid,
            metrics=metric_set(rmse, mae, rsq))

## Find Best Tuning Parameters
rf_bestTune <- rf_CV_results %>%
  select_best("rmse")


## Finalize the Workflow & fit it
rf_final_wf <-
  rf_wf %>%
  finalize_workflow(rf_bestTune) %>%
  fit(data=bikeTrain_log)

## Predict
rf_pred <- rf_final_wf %>%
  predict(new_data = bikeTest_new) %>%
  bind_cols(.,bikeTest) %>% # bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and predictions
  rename(count = .pred) %>% # rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>%
  mutate(count = pmax(0,count)) %>% #pointwise max of (0,prediction)
  mutate(count = ifelse(is.na(count), 0, count)) %>%
  mutate(datetime=as.character(format(datetime))) #needed for right format


vroom_write(rf_pred, "rf_pred.csv", delim = ',')


##################
# Stacked Models #
##################

# remember library(stacks)

# penalized regression recipe
stack_recipe <- recipe(count~., data=bikeTrain_log) %>%
  step_mutate(weather=factor(weather)) %>% #change weather to a factor
  step_mutate(season=factor(season)) %>% #change season to a factor
  #  step_time(datetime, features=c("hour", "minute")) %>%
  step_rm(datetime, weather) %>% # remove datetime
  step_zv(all_predictors()) %>% #removes zero-variance predictors
  step_dummy(all_nominal_predictors()) %>% #make dummy vars
  step_normalize(all_numeric_predictors()) #make mean 0, sd=1

## split data for CV
folds_stack <- vfold_cv(bikeTrain_log, v = K, repeats=1)

## control settings for stacking models
untunedModel <- control_stack_grid() #if tuning over a grid
tunedModel <- control_stack_resamples() #if not tuning a model

# set Penalized Regression model, workflow, and tuning grid
preg_model_stack <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")
preg_wf_stack <- workflow() %>%
  add_recipe(stack_recipe) %>% #using old recipe
  add_model(preg_model_stack)
preg_tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = L) ## L^2 total tuning possibilities

# Run the CV
preg_models <- preg_wf_stack %>%
  tune_grid(resamples=folds_stack,
            grid=preg_tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untunedModel) # including the control grid in the tuning ensures you can
                                    # call on it later in the stacked model


# Create other resampling objects with different ML algorithms to include in a stacked model
# linear regression
lin_reg_model_stack <-
  linear_reg() %>%
  set_engine("lm")
lin_reg_wf_stack <-
  workflow() %>%
  add_model(lin_reg_model_stack) %>%
  add_recipe(stack_recipe)
lin_reg_model_stack_resamples <- # fit this model with cross validation
  fit_resamples(
                lin_reg_wf_stack,
                resamples = folds_stack,
                metrics = metric_set(rmse, mae, rsq),
                control = tunedModel
  )


#Poisson
# create model
pois_model_stack <- poisson_reg() %>%
  set_engine("glm") 
# create workflow with recipe and model
pois_wf_stack <- workflow() %>%
  add_recipe(stack_recipe) %>%
  add_model(pois_model_stack)
pois_model_stack_resamples <- # fit this model with cross validation
  fit_resamples(
    pois_wf_stack,
    resamples = folds_stack,
    metrics = metric_set(rmse, mae, rsq),
    control = tunedModel
  )

#Regression Tree
regtree_mod_stack <- decision_tree(tree_depth = tune(), 
                             cost_complexity = tune(),
                             min_n=tune()) %>% 
  set_engine("rpart") %>%
  set_mode("regression")
regtree_wf_stack <- workflow() %>%
  add_recipe(stack_recipe) %>%
  add_model(regtree_mod_stack)
L <- 5
regtree_tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = L)

# Run CV regtree
regtree_models <- regtree_wf_stack %>%
  tune_grid(resamples=folds_stack,
            grid=regtree_tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untunedModel) # including the control grid in the tuning ensures you can
                                    # call on it later in the stacked model


# Random Forest
rf_mod_stack <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("regression")
rf_wf_stack <- workflow() %>%
  add_recipe(stack_recipe) %>%
  add_model(rf_mod_stack)
rf_tuning_grid_stack <- grid_regular(mtry(range = c(1,10)),
                               min_n(),
                               levels = L)
# Run CV random forest
rf_models <- rf_wf_stack %>%
  tune_grid(resamples=folds_stack,
            grid=rf_tuning_grid_stack,
            metrics=metric_set(rmse, mae, rsq),
            control = untunedModel)


#CREATE STACK
## Specify with models to include
my_stack <- stacks() %>%
  add_candidates(preg_models) %>%
  add_candidates(lin_reg_model_stack_resamples) %>%
  add_candidates(pois_model_stack_resamples)  %>%
  add_candidates(regtree_models) %>%
  add_candidates(rf_models)

## Fit the stacked model
stack_mod <- my_stack %>%
  blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members() ## Fit the members to the dataset

## If you want to build your own metalearner you'll have to do so manually
## using
stackData <- as_tibble(my_stack)

## Use the stacked data to get a prediction
stack_pred <- stack_mod %>% 
  predict(new_data=bikeTest_new) %>%
  bind_cols(.,bikeTest) %>% # bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and predictions
  rename(count = .pred) %>% # rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>%
  mutate(count = pmax(0,count)) %>% #pointwise max of (0,prediction)
  mutate(count = ifelse(is.na(count), 0, count)) %>%
  mutate(datetime=as.character(format(datetime))) #needed for right format


vroom_write(stack_pred, "stack_pred.csv", delim = ',')



##########
## Bart ##
##########

bikeTrain <- vroom("./STAT348/KaggleBikeShare/bike-sharing-demand/train.csv")
bikeTest <- vroom("./STAT348/KaggleBikeShare/bike-sharing-demand/test.csv")

#Data Cleaning
#change each instance (only a single instance) of 4 to 3.
#bikeTrain$weather <- ifelse(bikeTrain$weather == 4, 3, bikeTrain$weather)
#remove casual and registered columns
bikeTrain <- bikeTrain[, !(names(bikeTrain) %in% c("casual", "registered"))]
#bikeTest$weather <- ifelse(bikeTest$weather == 4, 3, bikeTest$weather)

# transform to log
bikeTrain_log <- bikeTrain %>%
  mutate(count = log(count)) %>%
  mutate(time = as.factor(as.integer(format(bikeTrain$datetime, "%H")))) %>%
  mutate(dayofweek = as.factor(format(bikeTrain$datetime, "%A"))) 

bikeTest_new <- bikeTest %>%
  mutate(time = as.factor(as.integer(format(bikeTest$datetime, "%H")))) %>%
  mutate(dayofweek = as.factor(format(bikeTest$datetime, "%A"))) 



bart_recipe <- recipe(count~., data=bikeTrain_log) %>%
  step_mutate(weather=factor(weather)) %>% #change weather to a factor
  step_mutate(season=factor(season)) %>% #change season to a factor
  step_time(datetime, features=c("hour", "minute")) %>%
  #step_rm(datetime, holiday) %>% # remove datetime
  step_zv(all_predictors()) %>% #removes zero-variance predictors
  step_dummy(all_nominal_predictors()) %>% #make dummy vars
  step_normalize(all_numeric_predictors()) #make mean 0, sd=1

bart_model <- bart(
  mode = "regression",
  engine = "dbarts",
  trees = 500,
  #prior_terminal_node_coef = NULL,
  #prior_terminal_node_expo = NULL,
  #prior_outcome_range = NULL
)
bart_wf <- workflow() %>%
  add_model(bart_model) %>%
  add_recipe(bart_recipe) %>%
  fit(data=bikeTrain_log)


bart_pred <- predict(bart_wf, new_data=bikeTest_new) %>%
  mutate(.pred = ifelse(is.na(.pred), 0, .pred)) %>%
  bind_cols(.,bikeTest) %>% # bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and predictions
  rename(count = .pred) %>% # rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>%
  mutate(count = pmax(0,count)) %>% #pointwise max of (0,prediction)
  mutate(count = ifelse(is.na(count), 0, count)) %>%
  mutate(datetime=as.character(format(datetime))) #needed for right format

vroom_write(bart_pred, "bart_pred.csv", delim = ',')
