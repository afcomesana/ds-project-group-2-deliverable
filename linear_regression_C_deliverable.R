### Developed by Moutushi Sen ###
# run_C_linear_regression_clean.R
# Cleaned, robust linear regression pipeline for C1/C2/C3
# - requires folders: Discharge, Meteo_csv, Rivers_csv, Meteo_csv writable

pkgs <- c("data.table","dplyr","lubridate")
for(p in pkgs) if(!requireNamespace(p, quietly=TRUE)) install.packages(p)
library(data.table); library(dplyr); library(lubridate)

# ===============================
#  OUTPUT DIRECTORY
# ===============================
output_dir <- "output/LM"

if (!dir.exists("output")) dir.create("output")
if (!dir.exists(output_dir)) dir.create(output_dir)

message("Saving all LM outputs into: ", output_dir)


# -------------------------
# metrics
# -------------------------
rmse <- function(a,b) sqrt(mean((a-b)^2, na.rm=TRUE))
mae  <- function(a,b) mean(abs(a-b), na.rm=TRUE)
NSE  <- function(obs, sim) {
  if(all(is.na(obs))) return(NA_real_)
  1 - sum((obs - sim)^2, na.rm=TRUE) / sum((obs - mean(obs, na.rm=TRUE))^2, na.rm=TRUE)
}

# -------------------------
# load discharge
# -------------------------
read_flow_safe <- function(path) {
  if(!file.exists(path)) stop("File not found: ", path)
  dt <- fread(path)
  # find a column named 'flow' case-insensitive
  idx <- which(tolower(names(dt)) == "flow")
  if(length(idx) >= 1) return(dt[[idx[1]]])
  # fallback: if exactly one numeric column, use it
  num_cols <- which(sapply(dt, is.numeric))
  if(length(num_cols) >= 1) return(dt[[num_cols[1]]])
  stop("No suitable 'flow' column found in: ", path)
}

C1 <- read_flow_safe("Discharge/discharge_C1.csv")
C2 <- read_flow_safe("Discharge/discharge_C2.csv")
C3 <- read_flow_safe("Discharge/discharge_C3.csv")
n <- length(C1)

# -------------------------
# load meteo (weekly aggregated)
# -------------------------
m1_file <- "Meteo_csv/C1_meteo.csv"
m2_file <- "Meteo_csv/C2_meteo.csv"
m3_file <- "Meteo_csv/C3_meteo.csv"
if(!file.exists(m1_file) || !file.exists(m2_file) || !file.exists(m3_file)) {
  stop("One or more Meteo_csv/C?_meteo.csv files missing. Check: ", m1_file, ", ", m2_file, ", ", m3_file)
}
m1 <- fread(m1_file); setnames(m1, tolower(names(m1)))
m2 <- fread(m2_file); setnames(m2, tolower(names(m2)))
m3 <- fread(m3_file); setnames(m3, tolower(names(m3)))

# -------------------------
# rivers: read and assemble with set() (avoid copying)
# -------------------------
river_files <- list.files("Rivers_csv", pattern="\\.csv$", full.names=TRUE)
if(length(river_files) == 0) warning("No CSVs found in Rivers_csv/")

# expected river names (use consistent names for output columns)
river_names_expected <- c("Arboga","Ballsta","Brobacken","Enkoping","Eskilstuna","Fyris",
                          "Hedstrommen","Kolback","Koping","Lovsta","Marsta","Orsund",
                          "Oxund","Racksta","Sagan","Sava","Svart")

# create empty data.table with preallocated columns to avoid shallow-copy warnings
rivers_dt <- data.table(index = seq_len(n))
for(rn in river_names_expected) rivers_dt[, (rn) := rep(NA_real_, n)]

# helper to find flow column in a river CSV
get_flow_vec_from_file <- function(fn, n) {
  rdat <- tryCatch(fread(fn), error=function(e) NULL)
  if(is.null(rdat)) return(rep(NA_real_, n))
  # pick column named 'flow' (case-insensitive) else second column else first numeric
  fi <- which(tolower(names(rdat)) == "flow")
  if(length(fi) >= 1) {
    v <- rdat[[fi[1]]]
  } else if (ncol(rdat) >= 2) {
    v <- rdat[[2]]
  } else {
    num_cols <- which(sapply(rdat, is.numeric))
    if(length(num_cols) >= 1) v <- rdat[[num_cols[1]]] else v <- rdat[[1]]
  }
  # ensure length n
  if(length(v) < n) v <- c(v, rep(NA_real_, n - length(v)))
  else if(length(v) > n) v <- v[1:n]
  as.numeric(v)
}

# populate rivers_dt by matching file names (case-insensitive)
for(fn in river_files) {
  bn <- tolower(basename(fn))
  for(rn in river_names_expected) {
    if(grepl(tolower(rn), bn, fixed = FALSE)) {
      vec <- get_flow_vec_from_file(fn, n)
      # set by column name in-place
      set(rivers_dt, j = rn, value = vec)
      break
    }
  }
}

# -------------------------
# grouped sums for north/west (use provided groupings)
# -------------------------
north_names <- c("Fyris","Orsund","Sava","Lovsta","Marsta","Oxund")
west_names  <- c("Koping","Hedstrommen","Arboga","Kolback","Eskilstuna","Svart")
# ensure columns exist (they do by preallocation)
rivers_dt[, north := rowSums(.SD, na.rm = TRUE), .SDcols = north_names]
rivers_dt[, west  := rowSums(.SD, na.rm = TRUE), .SDcols = west_names]

# -------------------------
# align lengths L and build df_mod
# -------------------------
L <- min(length(C1), length(C2), length(C3), nrow(m1), nrow(m2), nrow(m3))
cat("Using L =", L, "\n")

df_mod <- data.frame(
  C1 = C1[1:L],
  C2 = C2[1:L],
  C3 = C3[1:L],
  rivers_dt[1:L, c(river_names_expected, "north", "west"), with = FALSE],
  stringsAsFactors = FALSE
)

# attach meteo predictors keyed to each interface (safe indexing; if missing use NA)
safe_col <- function(dt, name, L) {
  name <- tolower(name)
  if(name %in% names(dt)) return(dt[[name]][1:L])
  return(rep(NA_real_, L))
}

df_mod$C1_wind_speed <- safe_col(m1, "wind_speed", L)
df_mod$C1_wind_dir   <- safe_col(m1, "wind_dir", L)
df_mod$C1_precip     <- if("precip" %in% names(m1)) safe_col(m1, "precip", L) else safe_col(m1, "precip_mm", L)

df_mod$C2_wind_speed <- safe_col(m2, "wind_speed", L)
df_mod$C2_wind_dir   <- safe_col(m2, "wind_dir", L)
df_mod$C2_precip     <- if("precip" %in% names(m2)) safe_col(m2, "precip", L) else safe_col(m2, "precip_mm", L)

df_mod$C3_wind_speed <- safe_col(m3, "wind_speed", L)
df_mod$C3_wind_dir   <- safe_col(m3, "wind_dir", L)
df_mod$C3_precip     <- if("precip" %in% names(m3)) safe_col(m3, "precip", L) else safe_col(m3, "precip_mm", L)

# -------------------------
# train/test split (robust)
# -------------------------
min_train <- 10; min_test <- 10
if (L < (min_train + min_test)) stop("Not enough rows for minimum train/test split.")

train_end <- max(min_train, floor(0.8 * L))
test_start <- train_end + 1
test_end <- L
if ((test_end - test_start + 1) < min_test) {
  test_start <- max(1, L - min_test + 1)
  train_end <- test_start - 1
  if (train_end < min_train) {
    train_end <- min_train
    test_start <- train_end + 1
    test_end <- L
  }
}
train_idx <- seq_len(train_end)
test_idx  <- seq(test_start, test_end)

cat("Train:", min(train_idx), "to", max(train_idx), " (", length(train_idx), " rows )\n")
cat("Test :", min(test_idx), "to", max(test_idx), " (", length(test_idx), " rows )\n")


#### --- add monthly dummies and save coefficient summaries for C1/C2/C3 --- ####

# install broom if needed for tidy()
if (!requireNamespace("broom", quietly = TRUE)) install.packages("broom")
library(broom)

# 1) create df_mod$date and df_mod$month if possible
# prefer the meteo file m1 (C1) date column if it exists
if ("date" %in% tolower(names(m1))) {
  # find the actual name (case-insensitive)
  date_col <- names(m1)[which(tolower(names(m1)) == "date")[1]]
  df_mod$date <- as.Date(m1[[date_col]][1:L])
} else {
  # fallback: if no real dates, create synthetic weekly dates starting at 2000-01-01
  # (adjust start date if you have a real one)
  df_mod$date <- seq.Date(from = as.Date("2000-01-01"), by = "week", length.out = nrow(df_mod))
}
# Add month factor (1-12)
df_mod$month <- factor(lubridate::month(df_mod$date))

# 2) update the predictor sets to include factor(month)
pred_C1 <- c("north","west","C1_wind_speed","C1_wind_dir","C1_precip","factor(month)")
pred_C2 <- c("north","west","C2_wind_speed","C2_wind_dir","C2_precip","factor(month)")
pred_C3 <- c("north","west","C3_wind_speed","C3_wind_dir","C3_precip","factor(month)")

# 3) modify fit_and_eval_lm (or create a wrapper) to accept formula directly.
# If you keep your existing function, call it with these predictor vectors.
# Example calling your fit function (assuming it accepts pred_cols):
results <- list()
results$C1 <- fit_and_eval_lm("C1", pred_C1)
results$C2 <- fit_and_eval_lm("C2", pred_C2)
results$C3 <- fit_and_eval_lm("C3", pred_C3)

# 4) Save coefficient tables and model summaries to output directory
output_dir <- "output/LM"   # change if you use a different output structure
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

save_model_outputs <- function(lm_object, name_prefix) {
  # tidy table (estimates, std.error, p.value)
  tidy_df <- broom::tidy(lm_object)
  fwrite(tidy_df, file = file.path(output_dir, paste0(name_prefix, "_coef_table.csv")))
  
  # confidence intervals
  conf_df <- as.data.frame(confint(lm_object))
  names(conf_df) <- c("2.5 %", "97.5 %")
  conf_df$term <- rownames(conf_df)
  setcolorder(as.data.table(conf_df), c("term","2.5 %","97.5 %"))
  fwrite(conf_df, file = file.path(output_dir, paste0(name_prefix, "_confint.csv")))
  
  # full summary text
  sum_txt_file <- file.path(output_dir, paste0(name_prefix, "_summary.txt"))
  capture.output(summary(lm_object), file = sum_txt_file)
}

# save coef tables for each model
save_model_outputs(results$C1$model, "C1_lm")
save_model_outputs(results$C2$model, "C2_lm")
save_model_outputs(results$C3$model, "C3_lm")

cat("Saved LM coefficient tables and summaries to:", output_dir, "\n")



# -------------------------
# helper: fit lm, predict, evaluate, save
# -------------------------
fit_and_eval_lm <- function(target, pred_cols) {
  form <- as.formula(paste(target, "~", paste(pred_cols, collapse = " + ")))
  lm_mod <- lm(form, data = df_mod[train_idx, , drop = FALSE])
  pred_vals <- predict(lm_mod, newdata = df_mod[test_idx, , drop = FALSE], interval = "confidence")
  pred_point <- as.numeric(pred_vals[,1])
  actual <- df_mod[test_idx, target]
  cat("----", target, "LM metrics ----\n")
  cat("RMSE:", rmse(actual, pred_point), " MAE:", mae(actual, pred_point), " NSE:", NSE(actual, pred_point), "\n")
  out_df <- data.frame(index = test_idx, actual = actual, pred = pred_point)
  fwrite(out_df, file = file.path(output_dir, paste0(target, "_lm_predictions.csv")))
  return(list(model = lm_mod, pred = pred_point, actual = actual))
}

# -------------------------
# run models for C1, C2, C3
# -------------------------
results <- list()
results$C1 <- fit_and_eval_lm("C1", c("north","west","C1_wind_speed","C1_wind_dir","C1_precip"))
results$C2 <- fit_and_eval_lm("C2", c("north","west","C2_wind_speed","C2_wind_dir","C2_precip"))
results$C3 <- fit_and_eval_lm("C3", c("north","west","C3_wind_speed","C3_wind_dir","C3_precip"))

# -------------------------
# derive percentages and evaluate
# -------------------------
p1 <- pmax(0, results$C1$pred); p2 <- pmax(0, results$C2$pred); p3 <- pmax(0, results$C3$pred)
denom <- p1 + p2 + p3
pred_pct1 <- ifelse(denom==0, 0, p1/denom); pred_pct2 <- ifelse(denom==0,0,p2/denom); pred_pct3 <- ifelse(denom==0,0,p3/denom)

act1 <- pmax(0, results$C1$actual); act2 <- pmax(0, results$C2$actual); act3 <- pmax(0, results$C3$actual)
denom_act <- act1 + act2 + act3
act_pct1 <- ifelse(denom_act==0,0,act1/denom_act); act_pct2 <- ifelse(denom_act==0,0,act2/denom_act); act_pct3 <- ifelse(denom_act==0,0,act3/denom_act)

cat("Percentage RMSEs (LM derived): C1", rmse(act_pct1,pred_pct1),
    "C2", rmse(act_pct2,pred_pct2),
    "C3", rmse(act_pct3,pred_pct3), "\n")

# save percentages
pct_df <- data.frame(index = test_idx,
                     act_pct1 = act_pct1, act_pct2 = act_pct2, act_pct3 = act_pct3,
                     pred_pct1 = pred_pct1, pred_pct2 = pred_pct2, pred_pct3 = pred_pct3)
fwrite(pct_df, file = file.path(output_dir, "C_lm_predicted_percentages.csv"))


# -------------------------
# per-interface feature files (for XGBoost later)
# -------------------------
df_C1 <- df_mod[, c("C1", river_names_expected, "north", "west",
                    "C1_wind_speed","C1_wind_dir","C1_precip")]
df_C2 <- df_mod[, c("C2", river_names_expected, "north", "west",
                    "C2_wind_speed","C2_wind_dir","C2_precip")]
df_C3 <- df_mod[, c("C3", river_names_expected, "north", "west",
                    "C3_wind_speed","C3_wind_dir","C3_precip")]

df_C1$index <- seq_len(nrow(df_C1)); df_C2$index <- seq_len(nrow(df_C2)); df_C3$index <- seq_len(nrow(df_C3))
fwrite(df_C1, file.path(output_dir, "df_C1_features.csv"))
fwrite(df_C2, file.path(output_dir, "df_C2_features.csv"))
fwrite(df_C3, file.path(output_dir, "df_C3_features.csv"))
cat("Saved per-interface feature files: Meteo_csv/df_C1_features.csv, df_C2_features.csv, df_C3_features.csv\n")

# -------------------------
# combined LM predictions (with percentages)
# -------------------------
lm_preds <- data.frame(index = test_idx,
                       C1_actual = results$C1$actual, C1_pred = results$C1$pred,
                       C2_actual = results$C2$actual, C2_pred = results$C2$pred,
                       C3_actual = results$C3$actual, C3_pred = results$C3$pred,
                       pred_pct1 = pred_pct1, pred_pct2 = pred_pct2, pred_pct3 = pred_pct3,
                       act_pct1 = act_pct1, act_pct2 = act_pct2, act_pct3 = act_pct3)
fwrite(lm_preds, file = file.path(output_dir, "C_lm_predictions_combined.csv"))
cat("Saved combined LM predictions: Meteo_csv/C_lm_predictions_combined.csv\n")
cat("LM predictions and percentages saved in Meteo_csv/\n")

# -------------------------
# done
# -------------------------
message("Script finished successfully.")

