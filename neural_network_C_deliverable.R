# run_C_nn.R
# ------------------------------------------
# Train simple MLP (nnet) for C1, C2, C3 using
# feature CSVs produced by the LM script.
#
# INPUT  (from LM):
#   output/LM/df_C1_features.csv
#   output/LM/df_C2_features.csv
#   output/LM/df_C3_features.csv
#
# OUTPUT (this script):
#   output/NN/
#     C1_nn_predictions.csv
#     C2_nn_predictions.csv
#     C3_nn_predictions.csv
#     C_nn_predicted_percentages.csv
#     C_nn_predictions_combined.csv
#     models/C1_nn.rds, C2_nn.rds, C3_nn.rds
#     nn_log.txt
# ------------------------------------------

pkgs <- c("data.table", "nnet")
for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
library(data.table)
library(nnet)

# ---------- metrics ----------
rmse <- function(a, b) sqrt(mean((a - b)^2, na.rm = TRUE))
mae  <- function(a, b) mean(abs(a - b), na.rm = TRUE)
R2   <- function(obs, pred) {
  1 - sum((obs - pred)^2, na.rm = TRUE) /
    sum((obs - mean(obs, na.rm = TRUE))^2, na.rm = TRUE)
}
NSE  <- R2   # for your report you can still call it Nash–Sutcliffe

# ---------- paths & folders ----------
project_root <- "."
lm_dir      <- file.path(project_root, "output", "LM")
nn_dir      <- file.path(project_root, "output", "NN")
models_dir  <- file.path(nn_dir, "models")

dir.create("output", showWarnings = FALSE)
dir.create(nn_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(models_dir, showWarnings = FALSE)

message("NN outputs will be saved in: ", nn_dir)

# feature files produced by LM script
f1 <- file.path(lm_dir, "df_C1_features.csv")
f2 <- file.path(lm_dir, "df_C2_features.csv")
f3 <- file.path(lm_dir, "df_C3_features.csv")

missing <- c(f1, f2, f3)[!file.exists(c(f1, f2, f3))]
if (length(missing) > 0) {
  stop(
    "Missing feature file(s) for NN:\n",
    paste(missing, collapse = "\n"),
    "\nRun the LM script first so df_C*_features.csv exist in output/LM/."
  )
}

# ---------- helper: prepare numeric matrix ----------
prepare_nn <- function(df, target_name) {
  # drop index if present
  if ("index" %in% names(df)) df$index <- NULL
  y <- df[[target_name]]
  X <- df[, setdiff(names(df), target_name), with = FALSE]
  
  # convert any factors to dummies, keep everything numeric
  Xmm <- model.matrix(~ . - 1, data = X)  # -1 removes intercept column
  list(X = Xmm, y = as.numeric(y), feature_names = colnames(Xmm))
}

# ---------- train/test split ----------
train_test_split <- function(N, train_frac = 0.8) {
  train_n   <- max(10, floor(train_frac * N))  # at least 10 train points
  train_idx <- seq_len(train_n)
  test_idx  <- seq(train_n + 1, N)
  if (length(test_idx) < 5) {
    # if almost no test left, shrink train a bit
    test_idx  <- seq(max(1, N - 9), N)
    train_idx <- seq_len(min(N - length(test_idx), train_n))
  }
  list(train = train_idx, test = test_idx)
}

# ---------- core function: train NN for a single basin ----------
train_nn_interface <- function(df, target_name, prefix) {
  prep <- prepare_nn(df, target_name)
  X <- prep$X
  y <- prep$y
  N <- nrow(X)
  
  if (N < 30) stop(prefix, ": Not enough rows (", N, ") to train NN.")
  
  split <- train_test_split(N, train_frac = 0.8)
  tr <- split$train
  te <- split$test
  
  # scale features using train stats
  scale_params <- list(
    mean = apply(X[tr, , drop = FALSE], 2, mean),
    sd   = apply(X[tr, , drop = FALSE], 2, sd)
  )
  Xs <- scale(X, center = scale_params$mean, scale = scale_params$sd)
  
  X_train <- Xs[tr, , drop = FALSE]
  X_test  <- Xs[te, , drop = FALSE]
  y_train <- y[tr]
  y_test  <- y[te]
  
  # small MLP; you can tune size, decay, maxit
  nn_mod <- nnet(
    x = X_train, y = y_train,
    size = 6,        # number of hidden units
    linout = TRUE,   # regression
    maxit = 1000,
    decay = 1e-4,
    trace = FALSE
  )
  
  pred_test <- as.numeric(predict(nn_mod, X_test))
  
  # metrics
  rmse_v <- rmse(y_test, pred_test)
  mae_v  <- mae(y_test, pred_test)
  r2_v   <- R2(y_test, pred_test)
  nse_v  <- NSE(y_test, pred_test)
  
  cat(sprintf(
    "%s NN: RMSE = %.4f  MAE = %.4f  R2 = %.4f  NSE = %.4f  (test n = %d)\n",
    prefix, rmse_v, mae_v, r2_v, nse_v, length(te)
  ))
  
  # save model object (RDS) + scale params
  saveRDS(
    list(model = nn_mod, scale = scale_params, features = prep$feature_names),
    file = file.path(models_dir, paste0(prefix, "_nn.rds"))
  )
  
  preds_df <- data.table(
    index  = te,
    actual = y_test,
    pred   = pred_test
  )
  fwrite(preds_df, file = file.path(nn_dir, paste0(prefix, "_nn_predictions.csv")))
  
  list(
    preds = preds_df,
    rmse  = rmse_v,
    mae   = mae_v,
    R2    = r2_v,
    NSE   = nse_v
  )
}

# ---------- load feature tables from LM ----------
df1 <- fread(f1)
df2 <- fread(f2)
df3 <- fread(f3)

# targets are the first column in each df (C1, C2, C3)
target1 <- names(df1)[1]
target2 <- names(df2)[1]
target3 <- names(df3)[1]

cat("NN targets:", target1, target2, target3, "\n")

# ---------- train models ----------
res1 <- train_nn_interface(df1, target1, "C1")
res2 <- train_nn_interface(df2, target2, "C2")
res3 <- train_nn_interface(df3, target3, "C3")

# ---------- derive percentages from NN predictions ----------
p1 <- pmax(0, res1$preds$pred)
p2 <- pmax(0, res2$preds$pred)
p3 <- pmax(0, res3$preds$pred)

# We assume indices (test sets) are aligned by value; to be strict, merge on index:
all_idx <- Reduce(intersect, list(res1$preds$index, res2$preds$index, res3$preds$index))
p1 <- res1$preds[match(all_idx, res1$preds$index), pred]
p2 <- res2$preds[match(all_idx, res2$preds$index), pred]
p3 <- res3$preds[match(all_idx, res3$preds$index), pred]

p1[p1 < 0] <- 0; p2[p2 < 0] <- 0; p3[p3 < 0] <- 0
denom <- p1 + p2 + p3
pred_pct1 <- ifelse(denom == 0, 0, p1 / denom)
pred_pct2 <- ifelse(denom == 0, 0, p2 / denom)
pred_pct3 <- ifelse(denom == 0, 0, p3 / denom)

act1 <- res1$preds[match(all_idx, res1$preds$index), actual]
act2 <- res2$preds[match(all_idx, res2$preds$index), actual]
act3 <- res3$preds[match(all_idx, res3$preds$index), actual]
denom_act <- act1 + act2 + act3
act_pct1 <- ifelse(denom_act == 0, 0, act1 / denom_act)
act_pct2 <- ifelse(denom_act == 0, 0, act2 / denom_act)
act_pct3 <- ifelse(denom_act == 0, 0, act3 / denom_act)

# percentage RMSEs
cat("NN percentage RMSEs:",
    " C1", rmse(act_pct1, pred_pct1),
    " C2", rmse(act_pct2, pred_pct2),
    " C3", rmse(act_pct3, pred_pct3), "\n")

# save percentage table
pct_df <- data.table(
  index     = all_idx,
  act_pct1  = act_pct1,
  act_pct2  = act_pct2,
  act_pct3  = act_pct3,
  pred_pct1 = pred_pct1,
  pred_pct2 = pred_pct2,
  pred_pct3 = pred_pct3
)
fwrite(pct_df, file = file.path(nn_dir, "C_nn_predicted_percentages.csv"))

# combined predictions in same structure as LM/XGBoost
combined <- data.table(
  index     = all_idx,
  C1_actual = act1, C1_pred = p1,
  C2_actual = act2, C2_pred = p2,
  C3_actual = act3, C3_pred = p3,
  pred_pct1 = pred_pct1, pred_pct2 = pred_pct2, pred_pct3 = pred_pct3,
  act_pct1  = act_pct1,  act_pct2  = act_pct2,  act_pct3  = act_pct3
)
fwrite(combined, file = file.path(nn_dir, "C_nn_predictions_combined.csv"))

# small log file
logfile <- file.path(nn_dir, "nn_log.txt")
writeLines(
  c(
    paste("NN run at", Sys.time()),
    paste("C1 RMSE =", round(res1$rmse, 4), "R2 =", round(res1$R2, 4)),
    paste("C2 RMSE =", round(res2$rmse, 4), "R2 =", round(res2$R2, 4)),
    paste("C3 RMSE =", round(res3$rmse, 4), "R2 =", round(res3$R2, 4))
  ),
  con = logfile
)

cat("Neural Network finished. Outputs written to:", nn_dir, "\n")
