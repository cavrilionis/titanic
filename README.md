# Titanic data analysis using GridSearchCV and XGBoost

An example of classification in machine learning where we predict the likelihood of survival of a passenger aboard the Titanic.

This Jupyter notebook illustrates the following:

- CSV file import
- Count of missing values and distinct values for each column
- Exploratory data analysis
- Creation of new columns (aka feature engineering)
- Imputation of missing values
- One hot encoding of categorical columns
- Data partitioning using stratified sampling
- Hyperparameter tuning for several classifiers
- Model fitting
- Decision threshold tuning
- Model fit assessment
- Visualization of the trade-off between precision and recall

# My Kaggle competition submission

You can find this Jupyter notebook on Kaggle:

https://www.kaggle.com/cavrilionis/titanic-data-analysis-gridsearchcv-and-xgboost

Titanic competition on Kaggle:

https://www.kaggle.com/c/titanic

# Notes

Please note that results of XGBoost model used in this Jupyter notebook are not identical with the results on Kaggle, as explained here:

https://github.com/dmlc/xgboost/issues/310

# Choosing a classifier

1. Get the top 2 classifiers from `GridSearchCV` with the highest mean `roc_auc` score using `cv=5`
2. Select the classifier from step 1 above which has the lowest standard deviation of `roc_auc` score

# Choosing a decision threshold

1. Compute F1 score on the training partition for each threshold between 0 and 1 by 0.001
2. Select the threshold which minimises F1 score on the training partition

# Results of XGBoost

| Environment | learning_rate | gamma | colsample_bytree | max_depth | minchild_weight | Threshold | Training partition accuracy | Validation partition accuracy | Scoring data accuracy |
|:------------|--------------:|------:|-----------------:|----------:|----------------:|----------:|----------------------------:|------------------------------:|----------------------:|
| My Mac      | 0.20          | 0.3   | 0.7              | 5         | 1               | 0.401     | 0.8652                      | 0.8156                        | N/A                   |
| Kaggle      | 0.25          | 0     | 0.5              | 6         | 5               | 0.419     | 0.8272                      | 0.8324                        | 0.7560                |

# Areas for further work

- Impute missing values in `Age` using the distribution of known values instead of the mean
- Use `Deck` with missing value imputation
- Try k-fold validation insted of train/test split
- Try more classifiers in `GridSearchCV`
- Try `RandomizedSearchCV` instead of `GridSearchCV`

# Authors

* **Christos Avrilionis** - *Initial work* - [cavrilionis](https://github.com/cavrilionis)

See also the list of [contributors](https://github.com/cavrilionis/titanic/graphs/contributors) who participated in this project.

# License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](https://github.com/cavrilionis/titanic/blob/master/LICENSE) file for details
