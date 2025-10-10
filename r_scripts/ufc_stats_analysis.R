# Install packages (run only once)
#install.packages("devtools")    # allows installing packages directly from GitHub.
#install.packages("dplyr")       # for data manipulation (grouping, summarising, arranging, etc.)
#install.packages("ggplot2")     # for data visualization

# Install ufc.stats from GitHub (run only once)
#devtools::install_github("mtoto/ufc.stats")

# Load packages
library(ufc.stats)
library(dplyr)
library(ggplot2)

# Load the dataset
data("ufc_stats")

dataset_information <- function() {
  cat("Dataset Information:\n")
  cat("Number of rows:", nrow(ufc_stats), "\n")
  cat("Number of columns:", ncol(ufc_stats), "\n")
  print(colnames(ufc_stats))
  print(str(ufc_stats))
  #print(summary(ufc_stats))
}

dataset_information()

top_fighters <- function(stat = "significant_strikes_landed", n = 10) {
  ufc_stats %>%
    group_by(fighter) %>%
    summarise(total = sum(.data[[stat]], na.rm = TRUE)) %>%
    arrange(-total) %>%
    head(n)
}
top_fighters("significant_strikes_landed", 5)

ufc_stats %>%
  group_by(fighter) %>%
  summarise(total_strikes = sum(significant_strikes_landed, na.rm = TRUE)) %>%
  arrange(-total_strikes) %>%
  head(10) %>%
  ggplot(aes(x = reorder(fighter, total_strikes), y = total_strikes)) +
  geom_col(fill = "firebrick") +
  coord_flip() +
  labs(title = "Top 10 UFC Fighters by Significant Strikes",
       x = "Fighter", y = "Total Significant Strikes")

colnames(ufc_stats)

#setwd("C:/Projects/UFC")
#getwd()
#write.csv(ufc_stats, "ufc_stats.csv", row.names = FALSE)
#getwd()
#save_csv <- function(file = "ufc_stats.csv") {
  #write.csv(cleaned_ufc_stats, "data/ufc_stats_clean.csv", row.names = FALSE)
#}

ufc_stats %>%
  filter(fighter %in% c("Conor McGregor", "Khabib Nurmagomedov")) %>%
  group_by(fighter) %>%
  summarise(avg_strikes = mean(significant_strikes_landed, na.rm = TRUE))
