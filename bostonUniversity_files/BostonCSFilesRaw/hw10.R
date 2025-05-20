# Load required packages
if (!require("knitr")) install.packages("knitr")
library(knitr)

# Sample probability distribution
outcomes <- c(0, 1, 2, 3, 4, 5, 6, 7)
probabilities <- c(0.4, 0.28, 0.16, 0.05, 0.04, 0.03, 0.02, 0.02)

# Print outcomes and probabilities as a table
print("Outcomes and Their Probabilities:")
kable(data.frame(Outcome = outcomes, Probability = probabilities), format = "markdown")

# Histogram data
hist_data <- rep(outcomes, times = round(probabilities * 100))

# Save histogram with line to a PNG file
png("images/probability_histogram-HW10-METCS544.png", width = 700, height = 500)
hist(hist_data, 
     breaks = length(outcomes), 
     col = "skyblue", 
     main = "Histogram of the Probability Distribution", 
     xlab = "Outcomes", 
     ylab = "Frequency", 
     probability = TRUE)
lines(outcomes, probabilities, type = "b", pch = 19, col = "red")
dev.off()

# Calculating the mean
mean_value <- sum(outcomes * probabilities)
print(paste("Mean value:", mean_value))

# Calculating the standard deviation
std_dev <- sqrt(sum((outcomes - mean_value)^2 * probabilities))
print(paste("Standard Deviation:", std_dev))

# --- Second Problem: Expected Profit ---
profits <- c(-90000, 70000, 200000)
profit_probs <- c(0.40, 0.45, 0.15)
expected_profit <- sum(profits * profit_probs)
print(paste("Expected Profit:", expected_profit))

# --- Third Problem: Expected Winnings ---
winnings <- c(50, 10, -5)
winnings_probs <- c(0.0556, 0.1667, 0.7777)
expected_winnings <- sum(winnings * winnings_probs)
print(paste("Expected Winnings:", expected_winnings))