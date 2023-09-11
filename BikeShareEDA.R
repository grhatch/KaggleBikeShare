##
##  Bike Share EDA Code
##

## Libraries
library(tidyverse)
library(vroom) # USE THIS TO READ IN DATA; it's fast, and does a lot of things automatically
library(patchwork) # add/divide ggplots to show them in panels in the output
library(ggplot2)

## Read in the Data
bike <- vroom("./bike-sharing-demand/train.csv")

View(bike)

glimpse(bike)
skimr::skim(bike)
DataExplorer::plot_intro(bike)
DataExplorer::plot_correlation(bike)
DataExplorer::plot_bar(bike)
DataExplorer::plot_histogram(bike)


tempCount <- ggplot(data=bike, aes(x=temp, y=count)) +
  geom_point() +
  geom_smooth()

humidityCount <- ggplot(data=bike, aes(y=count, x=humidity)) +
  geom_point() +
  geom_smooth()

season <- ggplot(data=bike, aes(x=factor(season), y=count)) +
  geom_boxplot()

weather <- ggplot(data=bike, aes(x=factor(weather), y=count)) +
  geom_boxplot()


(season + weather) / (tempCount + humidityCount)
