# ufc_cli_args.R
cat("Starting UFC Stats CLI script...\n")

args <- commandArgs(trailingOnly = TRUE)

# ---- Parse arguments ----
# choice: 1=Top Strikers, 2=Top Grapplers, 3=Top Knockdowns, 4=Top Submission Artists, 5=Compare Fighters, 6=Exit
choice <- as.integer(args[1])
n <- ifelse(length(args) >= 2, as.integer(args[2]), 10)
f1 <- ifelse(length(args) >= 3, args[3], NA)
f2 <- ifelse(length(args) >= 4, args[4], NA)

# ---- Load packages and data ----
library(dplyr)
library(ggplot2)
library(ufc.stats)

data("ufc_stats")

# ---- Functions ----
top_fighters <- function(stat, n, file) {
  df <- ufc_stats %>%
    group_by(fighter) %>%
    summarise(total = sum(.data[[stat]], na.rm = TRUE)) %>%
    arrange(-total)
  
  # Print top-n to console
  print(head(df, n))
  
  # Plot top-n and save if file specified
  p <- ggplot(df[1:n, ], aes(x = reorder(fighter, total), y = total)) +
    geom_col(fill = "firebrick") +
    coord_flip() +
    labs(title = paste("Top", n, "UFC Fighters by", stat),
         x = "Fighter", y = paste("Total", stat))
  
  if (!is.null(file)) {
    ggsave(file, plot = p, width = 8, height = 6)
    cat("Plot saved to", file, "\n")
  }
}

compare_fighters <- function(f1, f2) {
  df <- ufc_stats %>%
    filter(fighter %in% c(f1, f2)) %>%
    group_by(fighter) %>%
    summarise(
      avg_strikes = mean(significant_strikes_landed, na.rm = TRUE),
      avg_takedowns = mean(takedown_successful, na.rm = TRUE),
      avg_submissions = mean(submission_attempt, na.rm = TRUE)
    )
  print(df)
}

# ---- Ensure output folder exists ----
out <- "outputs"
if (!dir.exists(out)) dir.create(out)

# ---- Execute choice ----
if (choice == 1) {
  str_choice <- "significant_strikes_landed"
  top_fighters(str_choice, n, file = paste0(out, "/top_strikers_", n, ".png"))
} else if (choice == 2) {
  str_choice <- "takedown_successful"
  top_fighters(str_choice, n, file = paste0(out, "/top_takedown_artists_", n, ".png"))
} else if (choice == 3) {
  str_choice <- "knockdowns"
  top_fighters(str_choice, n, file = paste0(out, "/top_knockdowns_", n, ".png"))
} else if (choice == 4) {
  str_choice <- "submission_attempt"
  top_fighters(str_choice, n, file = paste0(out, "/top_submission_artists_", n, ".png"))
} else if (choice == 5 && !is.na(f1) && !is.na(f2)) {
  compare_fighters(f1, f2)
} else if (choice == 6) {
  cat("Exiting...\n")
} else {
  cat("Invalid choice or missing arguments.\n")
}
