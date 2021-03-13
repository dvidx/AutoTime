#!/usr/bin/env Rscript

# Visual.R
# * Summary


# Todo
# - Create as a report?
# - Remove filter
# - Y Scale values based on actual values


# * Libraries
extrafont::loadfonts(device = "win")
library(ggplot2)
library(tidyverse)
library(dplyr)
library(hrbrthemes)
library(forecast)
library(lubridate)
library(viridis)
library(jcolors)
#library(optparse) # https://www.r-bloggers.com/2015/09/passing-arguments-to-an-r-script-from-command-lines/


# * Fetch Arguments
args = commandArgs(trailingOnly = TRUE)
# test if there is at least one argument: if not, return an error
if (length(args) == 0) {
  stop("X arguments must be supplied (input file).n", call. = FALSE)
}


# * Settings
# Set working directory
setwd(args[1])

# set timezone to deal with converting issue
Sys.setenv(TZ = "GMT")

# Set Theme
theme_set(theme_bw(base_size = 12, base_family = "Roboto Condensed"))
jcolors('default')
# display_all_jcolors_contin()
# display_all_jcolors()

n_lag = args[2] # Change according to model lag
n_lag = n_lag + 1
n_pred = args[3] # Change according to model forecast
n_pred = n_pred - 1

increment = args[5]



# * Read in Data
# ToDo - Combine forecast and test data beforehand and read in only one file
file_path = file.path(args[1], "1_prediction.csv")

# Read in test data (original data)
test_result = read.csv("./data/processed/old/test_solar.csv") 
# Convert Timestamp
test_result$Datetime = ymd_hms(test$Datetime)

# # Read in result (prediction)
# result = read.csv(file_path, header = FALSE) # ToDo - Remove and replace above

# ToDo - Should not be neede with above change
## Adjust lag and forecast

# if (n_lag > 1) {
#   for (i in 1:n_lag) {
#     result = result %>%
#       add_row(V1 = 0, .before = 1)
#   }
# }

# if (n_pred != 0) {
#   for (i in 1:n_pred) {
#     result = result %>%
#       add_row(V1 = 0, .after = nrow(result))     
#   }
# }

# # combine them in new dataset
# test_result = test
# test_result$result = result$V1


# * Calculate Errors and additional Information
# get time of day in hours
test_result$Time = strftime(test_result$Datetime, format = "%H:%M:%S", tz = "UTC")
test_result$Time = hms::as_hms(test_result$Time)
test_result$hour = hour(hms(test_result$Time))
test_result$Datetime = as.POSIXct(test_result$UNIXTime, origin = "1970-01-01") #?

# ToDo - Add calculating Night Time


# set all negative predictions to 0 (We know can't be lower than 0)
test_result$result[test_result$result < 0] = 0
# set night obs. to 0
test_result$result[test_result$Night == 0] = 0
test_result$Power[test_result$Night == 0] = 0

# ToDo - Automated check if values are incremental
# Adjust value if their continous but incremental
if (increment == TRUE) {
  # set values to increments of 0.06
  real_seq = seq(0, 30, by = 0.06)
  index = findInterval(test_result$result, real_seq)
  test_result$results2 = real_seq[index]
  test_result$result = test_result$results2 + 0.06
}


# Absolute error
test_result$diff_abs = abs(test_result$Power - test_result$result)
test_result$diff_abs = round(test_result$diff_abs, digits = 4)

# Percentage error
test_result$diff_per = (abs(test_result$Power - test_result$result) / test_result$Power) * 100
# Replace all infinite values with 0 
# (Maybe better option)
test_result$diff_per[!is.finite(test_result$diff_per)] = 0
test_result$diff_per = round(test_result$diff_per, digits = 4)

# True Error
test_result$error = test_result$result - test_result$Power


################ RECODE!!!!!!!!!!
#test_result=subset(test_result, test_result$Night == 1)

# ToDo - Think about proper way to output / report
# * Get Basic summary
summary(test_result)
# How many obs are above 5 % error
sum(test_result$diff_per > 5) / nrow(test_result) * 100
sum(test_result$diff_per > 5 & test_result$diff_abs > 0.15) / nrow(test_result) * 100

# Aggregate days absolute error
# This done to identify the max and min error day
# We then use these days for the plot to give a more fair represenation of the model
test_result$date = date(test_result$Datetime)

agg_days = test_result %>%
group_by(date) %>%
summarise(meanValue = mean(diff_abs))

day_max_error = agg_days$date[which.max(agg_days$meanValue)]
day_min_error = agg_days$date[which.min(agg_days$meanValue)]

# day_max_after = day_max_error %m+% days(1)
# day_min_after = day_min_error %m+% days(1)

# TEST PLOTS
###


# COMPARE results vs actual
plot1 = test_result %>%
ggplot(aes(x = Datetime)) +

geom_line(aes(y = result, color = "Predicted"),
            lwd = 1) +

geom_line(aes(y = Power, color = "Actual"),
            lwd = 1, alpha = 0.8) +

scale_colour_manual("",
                    breaks = c("Actual", "Predicted"),
                    values = c("#558aa6", "#BBBE64")) +

scale_x_datetime(limit = c(as.POSIXct(paste(day_max_error, "00:00:00")),
                          as.POSIXct(paste(day_max_error %m+% days(1), "00:00:00"))),
                  date_breaks = "6 hour",
                  expand = c(0, 0),
                  date_labels = "  %b %d \n %H:%M",
                  date_minor_breaks = "2 hours") +

scale_y_continuous(minor_breaks = seq(0, 30, 2.5)) +

labs(x = " Date", y = "Power", title = "Predicted vs. Actual Power") +

theme(
  axis.title = element_text(size = 15, face = 2),
  axis.text = element_text(size = 14, face = 2),
  axis.text.x = element_text(size = 12),
  legend.text = element_text(size = 12),
  plot.title = element_text(size = 18, hjust = 0.5, face = 4),
  panel.grid.major = element_line(color = "gray70", size = 0.25)
)

# Boxplot
plot2 = test_result %>%
filter(diff_per < 10) %>%
  ggplot(aes(x = "", y = diff_per)) +
    geom_boxplot() +
    labs(y = "Percentage error", x = "", title = "Error Boxplot") +
    theme(
      axis.title = element_text(size = 15, face = 2),
      axis.text = element_text(size = 14, face = 2),
      plot.title = element_text(size = 18, hjust = 0.5, face = 4),
      panel.grid.major = element_line(color = "gray70", size = 0.25)
    )



# Error distribution
plot3 = test_result %>%
filter(diff_per < 20) %>%
  ggplot(aes(x = diff_per)) +
    geom_density(fill = "#558aa6", color = "#558aa6", alpha = 0.8) +

    labs(x = "Percentage error", y = "Density", title = "Error distribution") +

    scale_x_continuous(minor_breaks = seq(0, 150, 5),
                        expand = c(0, 0)) +

    theme(
      axis.title = element_text(size = 15, face = 2),
      axis.text = element_text(size = 14, face = 2),
      plot.title = element_text(size = 18, hjust = 0.5, face = 4),
      panel.grid.major = element_line(color = "gray70", size = 0.25)
    )

# Histogram
plot4 = test_result %>%
filter(diff_per < 20) %>%
ggplot(aes(x = diff_per)) +
geom_histogram(colour = "#558aa6", fill = "#558aa6", binwidth = 0.5, alpha = 0.6, boundary = 0, closed = "left") +

stat_bin(binwidth = 0.5, geom = "text", aes(label = round(..count.. / nrow(test_result) * 100, 1)), vjust = -1, boundary = 0) +

labs(x = "Percentage error", y = "Observations", title = "Error distribution") +

scale_x_continuous(minor_breaks = seq(0, 20, 0.5)) +

scale_y_continuous(minor_breaks = seq(0, 4500, 500),
                    expand = c(0, 500)) +

theme(
  axis.title = element_text(size = 15, face = 2),
  axis.text = element_text(size = 14, face = 2),
  plot.title = element_text(size = 18, hjust = 0.5, face = 4),
  panel.grid.major = element_line(color = "gray70", size = 0.25),
)



# Percentage error relative to abs error
plot5 = test_result %>%
filter(diff_per < 150) %>%
filter(diff_abs < 2.5) %>%
ggplot(aes(x = diff_abs)) +
geom_point(aes(y = diff_per), color = "#558aa6") +
scale_y_continuous(breaks = seq(0, 150, 25),
                    minor_breaks = seq(0, 150, 5)) +
scale_x_continuous(minor_breaks = seq(0, 10, 0.5),
                    expand = c(0.01, 0)) +
labs(x = "Abolute error", y = "Percentage error", title = "Error comparison") +
theme(
  axis.title = element_text(size = 15, face = 2),
  axis.text = element_text(size = 14, face = 2),
  plot.title = element_text(size = 18, hjust = 0.5, face = 4),
  panel.grid.major = element_line(color = "gray70", size = 0.25)
  )




# Error relative to Power values
plot6 = test_result %>%
filter(diff_abs < 2.5) %>%
ggplot(aes(x = Power)) +
geom_point(aes(y = diff_abs, color = diff_per)) + # , size = diff_per
scale_x_continuous(breaks = seq(0, 30, 5),
                    minor_breaks = seq(0, 30, 1),
                    limits = c(0, 30),
                    expand = c(0.01, 0)) +
scale_y_continuous(minor_breaks = seq(0, 2.5, 0.1),
                    expand = c(0.01, 0)) +
labs(x = "Power", y = "Abolute error", title = "Error comparison", color = " Percentage \n Error") +
theme(
  axis.title = element_text(size = 15, face = 2),
  axis.text = element_text(size = 14, face = 2),
  plot.title = element_text(size = 18, hjust = 0.5, face = 4),
  panel.grid.major = element_line(color = "gray70", size = 0.25),
  legend.text = element_text(size = 11)
) +
scale_color_viridis(option = "D")


# Error based on Hour of the day

plot7 = test_result %>%
filter(diff_per < 100) %>%
ggplot(aes(x = reorder(hour, sort(hour)), y = diff_per, fill = hour)) +
geom_boxplot(show.legend = FALSE) +

labs(x = " Hour of the day", y = "Percentage Error", title = "Error based on daytime") +

theme(
  axis.title = element_text(size = 15, face = 2),
  axis.text = element_text(size = 14, face = 2),
  axis.text.x = element_text(size = 12),
  plot.title = element_text(size = 18, hjust = 0.5, face = 4),
  panel.grid.major = element_line(color = "gray70", size = 0.25)
) +
scale_y_continuous(minor_breaks = seq(0, 100, 5))


plot8 = test_result %>%
filter(diff_abs < 2.5) %>%
ggplot(aes(x = reorder(hour, sort(hour)), y = diff_abs, fill = hour)) +
geom_boxplot(show.legend = FALSE) +

labs(x = " Hour of the day", y = "Absolute Error", title = "Error based on daytime") +

theme(
  axis.title = element_text(size = 15, face = 2),
  axis.text = element_text(size = 14, face = 2),
  axis.text.x = element_text(size = 12),
  plot.title = element_text(size = 18, hjust = 0.5, face = 4),
  panel.grid.major = element_line(color = "gray70", size = 0.25)
)

# ToDo
# See if error usuall to small or to high for certain hour
# Actual > Prediction Error will be negative
# Actual < Prediction Error will be positive 

# The chart shows the in the middle of the day we generally predict to low
# This means we can adjust by adding and thereby shifting the error 


plot9 = test_result %>%
filter(test_result_adjust$error < 3, test_result_adjust$error > -3) %>%
ggplot(aes(x = reorder(hour, sort(hour)), y = error, fill = hour)) +

geom_boxplot(show.legend = FALSE) +

stat_summary(fun = mean, geom = "point", shape = 4, size = 2, color = "red", fill = "red") +

labs(x = " Hour of the day", y = "Error (Power)", title = "Error based on daytime") +

scale_y_continuous(breaks = seq(-3, 3, 0.5),
                    minor_breaks = seq(-3, 3, 0.1),
                    limits = c(-3, 3),
                    expand = c(0, 0)) +


theme(
  axis.title = element_text(size = 15, face = 2),
  axis.text = element_text(size = 14, face = 2),
  axis.text.x = element_text(size = 12),
  plot.title = element_text(size = 18, hjust = 0.5, face = 4),
  panel.grid.major = element_line(color = "gray70", size = 0.25)
)

# * Adjusted version

# get mean for hour and substract / add
test_result_adjust = test_result %>%
group_by(hour) %>%
mutate(result = result - mean(error)) %>%
ungroup() %>%
select(-hour)

test_result_adjust$hour = hour(hms(test_result_adjust$Time))
test_result_adjust$hour = as.factor(test_result_adjust$hour)
test_result_adjust = test_result_adjust %>% relocate(hour, .after = Time)

# calculate the percentage error and absolute error
# Absolute error
test_result_adjust$diff_abs = abs(test_result_adjust$Power - test_result_adjust$result)
test_result_adjust$diff_abs = round(test_result_adjust$diff_abs, digits = 4)

# Percentage
test_result_adjust$diff_per = (abs(test_result_adjust$Power - test_result_adjust$result) / test_result_adjust$Power) * 100
# Replace all infinite values with 0
test_result_adjust$diff_per[!is.finite(test_result_adjust$diff_per)] = 0
test_result_adjust$diff_per = round(test_result_adjust$diff_per, digits = 4)

# Error
test_result_adjust$error = test_result_adjust$result - test_result_adjust$Power

# ToDo - Report
summary(test_result_adjust)
# How many obs are above 5 % error
sum(test_result_adjust$diff_per > 5) / nrow(test_result_adjust) * 100
sum(test_result_adjust$diff_per > 5 & test_result_adjust$diff_abs > 0.15) / nrow(test_result_adjust) * 100


plot10 = test_result_adjust %>%
filter(test_result_adjust$error < 3, test_result_adjust$error > -3) %>%
ggplot(aes(x = reorder(hour, sort(hour)), y = error, fill = hour)) +

geom_boxplot(show.legend = FALSE) +

stat_summary(fun = mean, geom = "point", shape = 4, size = 2, color = "red", fill = "red") +

labs(x = " Hour of the day", y = "Error (Power)", title = "Error based on daytime (Adjusted)") +

scale_y_continuous(breaks = seq(-3, 3, 0.5),
                    minor_breaks = seq(-3, 3, 0.1),
                    limits = c(-3, 3),
                    expand = c(0, 0)) +

theme(
  axis.title = element_text(size = 15, face = 2),
  axis.text = element_text(size = 14, face = 2),
  axis.text.x = element_text(size = 12),
  plot.title = element_text(size = 18, hjust = 0.5, face = 4),
  panel.grid.major = element_line(color = "gray70", size = 0.25)
)


# Error based on Hour of the day

plot11 = test_result_adjust %>%
filter(diff_per < 100) %>%
ggplot(aes(x = reorder(hour, sort(hour)), y = diff_per, fill = hour)) +
geom_boxplot(show.legend = FALSE) +

labs(x = " Hour of the day", y = "Percentage Error", title = "Error based on daytime (Adjusted)") +

theme(
  axis.title = element_text(size = 15, face = 2),
  axis.text = element_text(size = 14, face = 2),
  axis.text.x = element_text(size = 12),
  plot.title = element_text(size = 18, hjust = 0.5, face = 4),
  panel.grid.major = element_line(color = "gray70", size = 0.25)
) +
scale_y_continuous(minor_breaks = seq(0, 100, 5))


plot12 = test_result_adjust %>%
filter(diff_abs < 2.5) %>%
ggplot(aes(x = reorder(hour, sort(hour)), y = diff_abs, fill = hour)) +
geom_boxplot(show.legend = FALSE) +

labs(x = " Hour of the day", y = "Absolute Error", title = "Error based on daytime (Adjusted)") +

theme(
  axis.title = element_text(size = 15, face = 2),
  axis.text = element_text(size = 14, face = 2),
  axis.text.x = element_text(size = 12),
  plot.title = element_text(size = 18, hjust = 0.5, face = 4),
  panel.grid.major = element_line(color = "gray70", size = 0.25)
)



l = list(plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10, plot11, plot12)
l


# ? ????
# Get the range of errors
test_result_adjust$Date = as.Date(test_result_adjust$Datetime)
day_agg = aggregate(list(test_result_adjust["diff_abs"], test_result_adjust["diff_per"]), by = test_result_adjust["Date"], mean)


for (i in 2:nrow(test_result_adjust)) {
  test_result_adjust$cum_error[i] = test_result_adjust$cum_error[i - 1] + test_result_adjust$error[i]
}
# ? ??? 

library("svglite")

ggsave(file = "./plots/plot1.svg", plot = plot1, width = 16, height = 10)
ggsave(file = "./plots/plot2.svg", plot = plot2, width = 16, height = 10)
ggsave(file = "./plots/plot3.svg", plot = plot3, width = 16, height = 10)
ggsave(file = "./plots/plot4.svg", plot = plot4, width = 16, height = 10)
ggsave(file = "./plots/plot5.svg", plot = plot5, width = 16, height = 10)
ggsave(file = "./plots/plot6-2.svg", plot = plot6, width = 16, height = 10)
ggsave(file = "./plots/plot7.svg", plot = plot7, width = 16, height = 10)
ggsave(file = "./plots/plot8.svg", plot = plot8, width = 16, height = 10)
ggsave(file = "./plots/plot9.svg", plot = plot9, width = 16, height = 10)
ggsave(file = "./plots/plot10.svg", plot = plot10, width = 16, height = 10)
ggsave(file = "./plots/plot11.svg", plot = plot11, width = 16, height = 10)
ggsave(file = "./plots/plot12.svg", plot = plot12, width = 16, height = 10)


# ToDo - Better way to output?
# ggsave(myplot, filename = paste("myplot", ID, ".svg", sep = ""))


##########
# Colors #

#00677F
#A6CAD8
#79AEBF
#5097AA
#007A93
#00566A
#004355

#####################
# GGPlot parameters #

# ylim(c(0, 50))
# theme(axis.text.x = element_text(angle = 50, size = 16, vjust = 0.5))
# expand_limits(x = 0, y = 0) # Force plot to start at origin
# cooord_equal() # same scaling of axis
# scale_y_continuous(label = function(x) {return(paste(x, "Degrees Fahrenheit"))})  # Function for label

# chic$season <- factor(chic$season, levels = c("Spring", "Summer",
#                                               "Autumn", "Winter"))

# ggplot(chic, aes(x = date, y = temp, color = factor(season))) +
#   geom_point() +
#   labs(x = "Year", y = "Temperature (�F)") +
#   scale_color_discrete("", labels = c("",""))