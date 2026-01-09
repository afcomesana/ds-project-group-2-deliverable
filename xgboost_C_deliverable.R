### Developed by Moutushi Sen ###
# run_C_xgboost.R
# Train xgboost for C1/C2/C3 using feature CSVs created by LM script.
# Writes models + predictions + percentages into output/LM/

pkgs <- c("data.table","xgboost","Matrix")
for (p in pkgs) if (!requireNamespace(p, quietly=TRUE)) install.packages(p)
library(data.table); library(xgboost);

# ---- Paths (EDIT if needed) ----
project_root <- "."        # keep '.' if you're in project root (getwd() should show Documents/... earlier)
output_dir   <- file.path(project_root, "output", "XGBOOST")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
models_dir <- file.path(output_dir, "models")
dir.create(models_dir, showWarnings = FALSE)

# ---- required files ----
f1 <- file.path(output_dir, "df_C1_features.csv")
f2 <- file.path(output_dir, "df_C2_features.csv")
f3 <- file.path(output_dir, "df_C3_features.csv")

missing <- c(f1,f2,f3)[!file.exists(c(f1,f2,f3))]
if (length(missing)>0) {
  stop("Missing required feature file(s):\n", paste(missing, collapse="\n"),
       "\n\nRun the linear regression script or move the files to output/LM/")
}

# ---- helpers ----
rmse <- function(a,b) sqrt(mean((a-b)^2, na.rm=TRUE))
mae  <- function(a,b) mean(abs(a-b), na.rm=TRUE)

# ---- load features ----
df1 <- fread(f1); df2 <- fread(f2); df3 <- fread(f3)

# identify target column names (C1/C2/C3)
# assume first column is the target (as written by LM script)
target1 <- names(df1)[1]
target2 <- names(df2)[1]
target3 <- names(df3)[1]
cat("Targets:", target1, target2, target3, "\n")

# small function to prepare data for xgboost
prepare <- function(df, target_name) {
  # drop index if exists
  if ("index" %in% names(df)) df$index <- NULL
  y <- df[[target_name]]
  X <- df[, setdiff(names(df), target_name), with = FALSE]
  # convert to numeric matrix
  Xmat <- as.matrix(X)
  return(list(X = Xmat, y = y, feature_names = colnames(Xmat)))
}

# train/test split: use same train/test ratio used in LM script (it saved test indices as index in lm predictions)
# If you want to use the entire file you can adjust
# We'll use 80/20 split similar to LM
train_frac <- 0.8

train_test_split <- function(N, train_frac=0.8) {
  train_n <- floor(train_frac * N)
  train_idx <- seq_len(train_n)
  test_idx <- seq(train_n + 1, N)
  list(train = train_idx, test = test_idx)
}

# ---- Train per interface models ----
train_and_save <- function(df, target_name, outprefix) {
  prep <- prepare(df, target_name)
  N <- nrow(prep$X)
  if (N < 30) stop("Not enough rows to train xgboost")
  sp <- train_test_split(N, train_frac)
  dtrain <- xgb.DMatrix(prep$X[sp$train, , drop=FALSE], label = prep$y[sp$train])
  dtest  <- xgb.DMatrix(prep$X[sp$test, , drop=FALSE],  label = prep$y[sp$test])
  
  params <- list(objective = "reg:squarederror", eval_metric = "rmse",
                 eta = 0.05, max_depth = 6, subsample = 0.8, colsample_bytree = 0.7)
  
  # cross-validated rounds
  cv <- xgb.cv(params = params, data = dtrain, nrounds = 1000, nfold = 5,
               early_stopping_rounds = 30, verbose = 0, showsd = TRUE)
  best_iter <- cv$best_iteration
  cat(outprefix, "best_iter =", best_iter, "\n")
  model <- xgb.train(params = params, data = dtrain, nrounds = best_iter, verbose = 0)
  
  pred_test <- predict(model, dtest)
  # evaluation
  rmse_v <- rmse(prep$y[sp$test], pred_test)
  mae_v  <- mae(prep$y[sp$test], pred_test)
  cat(sprintf("%s: RMSE=%.4f MAE=%.4f (test rows=%d)\n", outprefix, rmse_v, mae_v, length(sp$test)))
  
  # save model and predictions
  model_file <- file.path(models_dir, paste0(outprefix, "_xgb.model"))
  xgb.save(model, model_file)
  
  preds_df <- data.table(index = sp$test,
                         actual = prep$y[sp$test],
                         pred   = pred_test)
  fwrite(preds_df, file = file.path(output_dir, paste0(outprefix, "_xgb_predictions.csv")))
  
  return(list(model = model, preds = preds_df, feature_names = prep$feature_names))
}

res1 <- train_and_save(df1, target1, "C1")
res2 <- train_and_save(df2, target2, "C2")
res3 <- train_and_save(df3, target3, "C3")

# ---- derive percentages from xgb predictions (align by index length) ----
# We assume each preds df has same number of rows and same ordering of test indices
p1 <- res1$preds$pred; p2 <- res2$preds$pred; p3 <- res3$preds$pred
p1[p1 < 0] <- 0; p2[p2 < 0] <- 0; p3[p3 < 0] <- 0
denom <- p1 + p2 + p3
pred_pct1 <- ifelse(denom == 0, 0, p1/denom)
pred_pct2 <- ifelse(denom == 0, 0, p2/denom)
pred_pct3 <- ifelse(denom == 0, 0, p3/denom)

act1 <- res1$preds$actual; act2 <- res2$preds$actual; act3 <- res3$preds$actual
denom_act <- act1 + act2 + act3
act_pct1 <- ifelse(denom_act == 0, 0, act1/denom_act)
act_pct2 <- ifelse(denom_act == 0, 0, act2/denom_act)
act_pct3 <- ifelse(denom_act == 0, 0, act3/denom_act)

pct_df <- data.table(index = res1$preds$index,
                     act_pct1 = act_pct1, act_pct2 = act_pct2, act_pct3 = act_pct3,
                     pred_pct1 = pred_pct1, pred_pct2 = pred_pct2, pred_pct3 = pred_pct3)

fwrite(pct_df, file = file.path(output_dir, "C_xgb_predicted_percentages.csv"))

# combined predictions for downstream plotting (same structure LM uses)
combined <- data.table(index = res1$preds$index,
                       C1_actual = act1, C1_pred = p1,
                       C2_actual = act2, C2_pred = p2,
                       C3_actual = act3, C3_pred = p3,
                       pred_pct1 = pred_pct1, pred_pct2 = pred_pct2, pred_pct3 = pred_pct3,
                       act_pct1 = act_pct1, act_pct2 = act_pct2, act_pct3 = act_pct3)
fwrite(combined, file = file.path(output_dir, "C_xgb_predictions_combined.csv"))

# small log
logfile <- file.path(output_dir, "xgb_log.txt")
writeLines(c(paste("xgboost run:", Sys.time()),
             paste("C1 model:", file.path(models_dir, "C1_xgb.model")),
             paste("C2 model:", file.path(models_dir, "C2_xgb.model")),
             paste("C3 model:", file.path(models_dir, "C3_xgb.model"))), con = logfile)

cat("XGBoost finished. Outputs written to:", output_dir, "\n")

# ---- Paths (EDITED TO SAVE IN XGBOOST FOLDER) ----
project_root <- "."  

output_dir   <- file.path(project_root, "output", "XGBOOST")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

models_dir <- file.path(output_dir, "models")
dir.create(models_dir, showWarnings = FALSE)

log_file <- file.path(output_dir, "xgb_log.txt")


