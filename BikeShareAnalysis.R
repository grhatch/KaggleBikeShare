library(tidyverse)
library(vroom)
library(tidymodels)

bikeTrain <- vroom("./STAT348/KaggleBikeShare/bike-sharing-demand/train.csv")
bikeTest <- vroom("./STAT348/KaggleBikeShare/bike-sharing-demand/test.csv")

View(bikeTrain)
View(bikeTest)

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