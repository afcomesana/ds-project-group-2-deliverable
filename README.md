Contains a python program for predicting flows at basin interfaces of Mälaren, and an additional R example which can be modified with new inputs if preferred.

The models make use of linear regresion to predict flows, and the features are river inflows to the lake and meteorological data at the interface(s) of interest, averaged over the same time span, for example weekly.

If the models are to be trained, a history of the interface flows from a hydrological model must be provided, averaged over the same time span as the features.

Additional options are included to group geographically close river inflows, for more interpretable results with less features.

main.py can be run in terminal, and the script contains an instruction of the options.

linear_regression_deliverable.R contains an example regression in R with explanatory comments, in case one prefers working in R and test feature selection there.

linear_regression_C_deliverable.R is an alternate version for regression in R.
