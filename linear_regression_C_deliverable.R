### Developed by Moutushi Sen ###
# Linear regression for C1/C2/C3
# WITH + WITHOUT rivers
# Includes: temperature + lagged precipitation + seasonality

pkgs <- c("data.table","dplyr","lubridate","broom")
for(p in pkgs) if(!requireNamespace(p, quietly=TRUE)) install.packages(p)
library(data.table); library(dplyr); library(lubridate); library(broom)

# ===============================
# OUTPUT DIRECTORY
# ===============================
output_dir <- "output/LM"
dir.create("output", showWarnings = FALSE)
dir.create(output_dir, showWarnings = FALSE)

# ===============================
# Metrics
# ===============================
rmse <- function(a,b) sqrt(mean((a-b)^2, na.rm=TRUE))
mae  <- function(a,b) mean(abs(a-b), na.rm=TRUE)
NSE  <- function(obs, sim){
  1 - sum((obs-sim)^2, na.rm=TRUE) /
      sum((obs-mean(obs, na.rm=TRUE))^2, na.rm=TRUE)
}

# ===============================
# Load discharge
# ===============================
read_flow_safe <- function(path){
  dt <- fread(path)
  idx <- which(tolower(names(dt))=="flow")
  if(length(idx)>0) return(dt[[idx[1]]])
  num <- which(sapply(dt,is.numeric))
  dt[[num[1]]]
}

C1 <- read_flow_safe("Discharge/discharge_C1.csv")
C2 <- read_flow_safe("Discharge/discharge_C2.csv")
C3 <- read_flow_safe("Discharge/discharge_C3.csv")

# ===============================
# Load meteo
# ===============================
m1 <- fread("Meteo_csv/C1_meteo.csv"); setnames(m1,tolower(names(m1)))
m2 <- fread("Meteo_csv/C2_meteo.csv"); setnames(m2,tolower(names(m2)))
m3 <- fread("Meteo_csv/C3_meteo.csv"); setnames(m3,tolower(names(m3)))

L <- min(length(C1), length(C2), length(C3), nrow(m1), nrow(m2), nrow(m3))

# ===============================
# Create lagged precipitation (KEY FIX)
# ===============================
lag1 <- function(x) c(NA, head(x, -1))

m1$precip_lag <- lag1(m1$precip)
m2$precip_lag <- lag1(m2$precip)
m3$precip_lag <- lag1(m3$precip)

# ===============================
# Rivers
# ===============================
river_names <- c("Arboga","Ballsta","Brobacken","Enkoping","Eskilstuna","Fyris",
                 "Hedstrommen","Kolback","Koping","Lovsta","Marsta","Orsund",
                 "Oxund","Racksta","Sagan","Sava","Svart")

rivers_dt <- data.table(matrix(0, nrow=L, ncol=length(river_names)))
setnames(rivers_dt, river_names)

for(f in list.files("Rivers_csv", full.names=TRUE)){
  bn <- tolower(basename(f))
  r <- fread(f)
  flow <- r[[which(tolower(names(r))=="flow")[1]]][1:L]
  for(nm in river_names)
    if(grepl(tolower(nm), bn)) rivers_dt[[nm]] <- flow
}

north <- c("Fyris","Orsund","Sava","Lovsta","Marsta","Oxund")
west  <- c("Koping","Hedstrommen","Arboga","Kolback","Eskilstuna","Svart")

rivers_dt$north <- rowSums(rivers_dt[, ..north])
rivers_dt$west  <- rowSums(rivers_dt[, ..west])

# ===============================
# Build modelling table
# ===============================
df_mod <- data.frame(
  C1=C1[1:L], C2=C2[1:L], C3=C3[1:L],
  rivers_dt[1:L,],
  C1_wind_speed=m1$wind_speed[1:L],
  C1_wind_dir  =m1$wind_dir[1:L],
  C1_temp      =m1$temp[1:L],
  C1_precip    =m1$precip[1:L],
  C1_precip_lag=m1$precip_lag[1:L],
  C2_wind_speed=m2$wind_speed[1:L],
  C2_wind_dir  =m2$wind_dir[1:L],
  C2_temp      =m2$temp[1:L],
  C2_precip    =m2$precip[1:L],
  C2_precip_lag=m2$precip_lag[1:L],
  C3_wind_speed=m3$wind_speed[1:L],
  C3_wind_dir  =m3$wind_dir[1:L],
  C3_temp      =m3$temp[1:L],
  C3_precip    =m3$precip[1:L],
  C3_precip_lag=m3$precip_lag[1:L]
)

df_mod$date  <- as.Date(m1$date[1:L])
df_mod$month <- factor(month(df_mod$date))

df_mod <- na.omit(df_mod)

# ===============================
# Train/test split
# ===============================
N <- nrow(df_mod)
tr <- 1:floor(0.8*N)
te <- (max(tr)+1):N

# ===============================
# Helper: fit + save
# ===============================
fit_lm <- function(target, preds){
  f <- as.formula(paste(target,"~",paste(preds,collapse="+")))
  m <- lm(f, df_mod[tr,])
  p <- predict(m, df_mod[te,])
  a <- df_mod[te,target]
  list(model=m, pred=p, actual=a, R2=NSE(a,p))
}

# ===============================
# Predictor sets (MATCH HENRIK)
# ===============================
pred_with <- list(
  C1=c("north","C1_temp","C1_precip","C1_precip_lag","factor(month)"),
  C2=c("north","west","C2_temp","C2_precip","C2_precip_lag","factor(month)"),
  C3=c("north","west","C3_temp","C3_precip","C3_precip_lag","factor(month)")
)

pred_wo <- list(
  C1=c("C1_temp","C1_precip","C1_precip_lag","factor(month)"),
  C2=c("C2_temp","C2_precip","C2_precip_lag","factor(month)"),
  C3=c("C3_temp","C3_precip","C3_precip_lag","factor(month)")
)

# ===============================
# Run models
# ===============================
res_with <- lapply(names(pred_with),
                   \(k) fit_lm(k, pred_with[[k]]))
res_wo   <- lapply(names(pred_wo),
                   \(k) fit_lm(k, pred_wo[[k]]))

# ===============================
# Save R² tables
# ===============================
fwrite(data.frame(Model="LM with rivers",
                  C1_R2=res_with$C1$R2,
                  C2_R2=res_with$C2$R2,
                  C3_R2=res_with$C3$R2),
       file.path(output_dir,"LM_R2_C_with_rivers.csv"))

fwrite(data.frame(Model="LM without rivers",
                  C1_R2=res_wo$C1$R2,
                  C2_R2=res_wo$C2$R2,
                  C3_R2=res_wo$C3$R2),
       file.path(output_dir,"LM_R2_C_without_rivers.csv"))

message("✅ Linear regression finished — temp + precip_lag INCLUDED")
