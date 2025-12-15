library(corrplot)
library(Hmisc)

# ------------------------
# --- Data Preparation ---
# ------------------------

# Needed data:
# - CSV files with flows at each interface (here weekly average)
# - CSV file with meteorological data on same timescale (also weekly)
# - Optional: River inflows on same timescale.

# -----------------------------
# --- Adding interface data ---
# -----------------------------

# This example concerns a basin with two entry-points, one to the north (A1) and one to the west (A2)
# For reference, A1 has coordinates coordinates xt = 658350, yt = 6595950
# And A2 coordinates xt = 652650, yt = 6592050,...,6590550
A1 <- read.csv("Discharge/discharge_A1.csv") 
A2 <- read.csv("Discharge/discharge_A2.csv") 

# IMPORTANT: Check on map and translate direction of flow relative to the basin of interest
# In this case, negative flow in the y-direction (i.e. south) at A1 corresponds to positive flow into the basin A
A1$flow <- -A1$flow
A2$flow <- A2$flow

# We are interested in finding the percentage of inflow from say A2
# Create percentage target
A1_p <- A1
A2_p <- A2
# If flow through interface is negative, set it to 0 as it contributes to no inflow
A1_p$flow[A1_p$flow < 0] <- 0
A2_p$flow[A2_p$flow < 0] <- 0
# Calculate percentage for interface of interest
A2_percentage <- A2_p$flow/(A1_p$flow+A2_p$flow)
# If both inflows are 0, replace resulting NaN with 0
A2_percentage[is.nan(A2_percentage)] <- 0

# Initiate data frame for each interface and target
df_A1 <- data.frame(A1)
df_A2 <- data.frame(A2)
df_A2p <- data.frame(A2_percentage)
# Add date to percentage target
df_A2p <- cbind(df_A1$date, df_A2p)

# --------------------------------------
# --- Optional: adding river inflows ---
# --------------------------------------

# Fetch river names from folder containing river inflows
file_names <- list.files(path = "Rivers_csv/")
if(length(file_names) == 0){
  stop("Empty or non-existing file_path")
}
file_paths <- file.path("./Rivers_csv", file_names)

# Add river flows as input to all data-frames
for (path in file_paths){
  R <- read.csv(path)
  df_A1 <- cbind(df_A1, R[2])
  df_A2 <- cbind(df_A2, R[2])
}

# Rename to actual river names
# Alphabetic order to match the order they were fetched
river_names <- c("Arboga", 
                 "Ballsta", 
                 "Brobacken", 
                 "Enkoping", 
                 "Eskilstuna", 
                 "Fyris", 
                 "Hedstrommen", 
                 "Kolback", 
                 "Koping", 
                 "Lovsta", 
                 "Marsta", 
                 "Orsund", 
                 "Oxund", 
                 "Racksta",
                 "Sagan",
                 "Sava",
                 "Svart")
names(df_A1) <- c("date", "A1", river_names)
names(df_A2) <- c("date", "A2", river_names)

# We can also add nearby rivers together for easier interpretation
# Here we use northern and western rivers relative to our basin 
# IMPORTANT: The rivers probably needs different splits for different interfaces

north <- c("Fyris", "Orsund", "Sava", "Lovsta", "Marsta", "Oxund")
west <- c("Koping", "Hedstrommen", "Arboga", "Kolback","Eskilstuna", "Svart", "Sagan", "Enkoping")
df_A1$north <- rowSums(df_A1[, north])
df_A1$west <- rowSums(df_A1[, west])

df_A2$north <- rowSums(df_A1[, north])
df_A2$west <- rowSums(df_A1[, west])


# ----------------------------------
# --- Adding meteorological data ---
# ----------------------------------

# In our case, the interfaces A1 and A2 are close in space so we can use the same weather info
# Otherwise one might consider loading one meteorological csv for each interface
meteo <- read.csv("Meteo_csv/A1_meteo2.csv")

# The only lagged feature we found correlating with the interface flow
# was rain from the previous week.

# Add lag of 1 week for rain
# Copy the rain column
precip_lag <- meteo[6]
names(precip_lag) <- c("precip_lag")
# Add na for first day and remove last day to create lag
new_row <- data.frame(precip_lag= NA)
precip_lag <- rbind(new_row, precip_lag)
precip_lag <- head(precip_lag, -1)

# Append lagged rain
meteo <- cbind(meteo, precip_lag)
# Append meteo data to each interface
df_A1 <- cbind(df_A1, meteo[2:7])
df_A2 <- cbind(df_A2, meteo[2:7])
# Remove first datapoint which doesn't have lag values
df_A1 <- na.omit(df_A1)
df_A2 <- na.omit(df_A2)

# We also add month to encode seasonality later
df_A1$month <- as.numeric(format(as.Date(as.Date(df_A1$date), format="%Y-m%-d%"),"%m"))
df_A2$month <- as.numeric(format(as.Date(as.Date(df_A2$date), format="%Y-m%-d%"),"%m"))


# ------------------------
# --- Helper functions ---
# ------------------------

# Plot predicted flow
plot_pred = function(preds, real, x, title){
  plot(x, real, type="l", lwd =1.5, xlab = "Time" , ylab = "Flow m3/s",  main = title)
  lines(x, preds[,1],type="l", lwd =1.5, col="red")
  #lines(x, preds[,2],lty="dashed", lwd =1.5, col="blue")
  #lines(x, preds[,3],lty="dashed", lwd =1.5, col="blue")
  #legend(x = "topright",          # Position
  #       legend = c("Actual data", "Predictions", "95% Confidence interval"),  # Legend texts
  #       lty = c(1, 1,1),           # Line types
  #       col = c("black","red", "blue"),           # Line colors
  #       lwd = 2)
  legend(x = "topright",          # Position
         legend = c("Actual data", "Predictions"),  # Legend texts
         lty = c(1, 1),           # Line types
         col = c("black","red"),           # Line colors
         lwd = 2)
}

#plot predicted percentage
plot_percentage = function(preds, real, x, title){
  plot(x, real, type="l", lwd =1.5, xlab = "Time" , ylab = "Percentage of flow",  main = title)
  lines(x, preds,type="l", lwd =1.5, col="red")
  legend(x = "topright",          # Position
         legend = c("Actual data", "Predictions"),  # Legend texts
         lty = c(1, 1),           # Line types
         col = c("black","red"),           # Line colors
         lwd = 2)
}

# ---------------
# --- A1 flow ---
# ---------------
# Reserve 20% (last five years, jan 2020 - nov 2024) as test data
A1_train <- df_A1[1:1043,]
A1_test <- df_A1[1044:1295,]

# Correlations and P-values for feature selections
rcorr(as.matrix(subset(A1_train, select = -date)))
# Correlation matrix
corrplot(cor(subset(A1_train, select = -date)), method='color')

# We can see that river inflows have the strongest correlations, but weather matters too
# A1 is a particularly easy example, as we can see a 1-to-1 correlation with the northern river inflow

# --------------------------------
# --- A1 predictions w. rivers ---
# --------------------------------
# Here using only the northern rivers as feature suffices
m_A1_rivers <- lm(A1 ~ north, data = A1_train)
summary(m_A1_rivers)
# From summary, we want R-squared close to 1,
# and statistically significant coefficents (Pr(>|t|) < 0.05)
# R-squared 0.9914

# Adding more features like weather leads to some statistically insignificant coefficents and not much improvement
m_A1_rivers_b <- lm(A1 ~ north + wind_speed + wind_dir + temp + precip + precip_lag + factor(month), data = A1_train)
summary(m_A1_rivers_b)
# R-squared 0.9923

# Predict on test data and plot
A1_prediction_rivers <- predict(m_A1_rivers, newdata = A1_test, interval = "confidence")
x = seq (1044:1295)
plot_pred(A1_prediction_rivers, A1_test$A1, x, "Jan 2020 - Nov 2024: Predicted vs. actual flow for A1 (using rivers)")
# Northern rivers give an almost perfect prediction

# ---------------------------------
# --- A1 predictions w/o rivers ---
# ---------------------------------
# Without river info, we have a harder task

# First we add a factor of month for seasonality
# The correlation between temperature and flow is not linear, as the hottest summer months have least flow
# The factored months will balance this out by negative weights for summer months
m_A1_no_rivers <- lm(A1 ~ wind_speed + wind_dir + temp + precip + precip_lag + factor(month), data = A1_train)
summary(m_A1_no_rivers)
# R-squared 0.4287
# We see a lot worse performance, but still above R-squared 0

A1_prediction_no_rivers <- predict(m_A1_no_rivers, newdata = A1_test, interval = "confidence")
x = seq (1044:1295)
plot_pred(A1_prediction_no_rivers, A1_test$A1, x, "Jan 2020 - Nov 2024: Predicted vs. actual flow for A1 (Only meteo data)")
# From the plot we see that weather data captures the seasonality pretty reasonably, but can't predict the peaks

# ---------------
# --- A2 flow ---
# ---------------
# Reserve 20% (last five years Jan 2020 - Nov 2024) as test data
A2_train <- df_A2[1:1043,]
A2_test <- df_A2[1044:1295,]

#Correlations and P-values
rcorr(as.matrix(subset(A2_train, select = -date)))
#Correlation matrix
corrplot(cor(subset(A2_train, select = -date)), method='color')
# Here the correlations are more diffuse than for A1
# But western and northern rivers are still important

# --------------------------------
# --- A2 predictions w. rivers ---
# --------------------------------
m_A2_rivers <- lm(A2 ~ north + west + wind_speed + wind_dir + temp + precip + precip_lag + factor(month), data = subset(A2_train, select = -date))
summary(m_A2_rivers)
# R-value 0.7824
# Temp and rain are not significant here and could be removed
m_A2_rivers_b <- lm(A2 ~ north + west + wind_speed + wind_dir + factor(month), data = subset(A2_train, select = -date))
summary(m_A2_rivers_b)
# Basically same R-value 0.7813

# Predict and plot
A2_prediction_rivers <- predict(m_A2_rivers, newdata = A2_test, interval = "confidence")
x = seq (1044:1295)
plot_pred(A2_prediction_rivers, A2_test$A2, x, "Jan 2020 - Nov 2024: Predicted vs. actual flow for A2 (Rivers + meteo)")
# The predictions look good

# --------------------------------
# --- A2 predictions w/o rivers ---
# --------------------------------
# Use all meteorological data available
m_A2_no_rivers <- lm(A2 ~ wind_speed + wind_dir + temp + precip + precip_lag + factor(month), data = A2_train)
summary(m_A2_no_rivers)
# Good coefficients
# R-value 0.4041 (Approximately the same as for A1 w/o rivers)

A2_prediction_no_rivers <- predict(m_A2_no_rivers, newdata = A2_test, interval = "confidence")
x = seq (1044:1295)
plot_pred(A2_prediction_no_rivers, A2_test$A2, x, "Jan 2020 - Nov 2024: Predicted vs. actual flow for A2 (Only meteo data)")
# Same as for A1, the meteorological model picks up the seasonal patterns but miss the peaks


# ------------------
# --- Percentage ---
# ------------------
A2p_train <- df_A2p[1:1043,]
A2p_test <- df_A2p[1044:1295,]

# Our predicted interface flows using rivers, changed to A2 percentage
pred1 <- A1_prediction_rivers[,1]
pred2 <- A2_prediction_rivers[,1]
pred1[pred1<0] <- 0
pred2[pred2<0] <- 0
pred_percentage <- pred2/(pred1+pred2)
pred_percentage[is.nan(pred_percentage)] <- 0

A2p_prediction_rivers <-pred_percentage
x = seq (1044:1295)
plot_percentage(A2p_prediction_rivers, A2p_test$A2_percentage, x, "Real vs. predicted percentages for A2 flow, using rivers")
# The actual data is very noisy, and if an interface flow changes between positive and negative 
# we can get sharp turns down to 0 percentage
# Our predictions miss the outliers, but manage to catch most up and downs in the "normal" range 60-90%

# Our predicted interface flows w/o rivers, changed to A2 percentage
pred1 <- A1_prediction_no_rivers[,1]
pred2 <- A2_prediction_no_rivers[,1]
pred1[pred1<0] <- 0
pred2[pred2<0] <- 0
pred_percentage <- pred2/(pred1+pred2)
pred_percentage[is.nan(pred_percentage)] <- 0

A2p_prediction_no_rivers <-pred_percentage
x = seq (1044:1295)
plot_percentage(A2p_prediction_no_rivers, A2p_test$A2_percentage, x, "Real vs. predicted percentages for A2 flow, w/o rivers")
# While the individual predictions for A1 and A2 were much worse without rivers, the percentage still seems to follow most of the seasonal pattern
# A1 seems to be underestimated with alot of false peaks for A2 percentage



