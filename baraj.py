import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor
import joblib


df = pd.read_csv("baraj.csv")

df.columns

plt.plot(df["GENERAL_DAM_OCCUPANCY_RATE"][:300])

plt.plot(df["GENERAL_DAM_RESERVED_WATER"][:300])


df["TimeStamp"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d")

df["Day"] = df["TimeStamp"].dt.day_name()

df["Month"] = df["TimeStamp"].dt.month_name()

df["Year"] = df["TimeStamp"].dt.year

drops = ["DATE","TimeStamp", "_id"]

df.columns

df.drop(labels = drops, axis = 1, inplace = True)

df.loc[(df["Month"] == "December") | (df["Month"] == "January") | (df["Month"] == "February")  , "Season"  ] = "Winter"

df.loc[(df["Month"] == "March") | (df["Month"] == "April") | (df["Month"] == "May"), "Season"  ] = "Spring"

df.loc[(df["Month"] == "June") | (df["Month"] == "July") | (df["Month"] == "August"), "Season"  ] = "Summer"

df.loc[(df["Month"] == "September") | (df["Month"] == "October") | (df["Month"] == "November"), "Season"  ] = "Autumn"

df.isnull().sum()

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
    
def grab_col_names(dataframe, cat_th=10, car_th=20):    
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car    

cat_cols, num_cols, cat_but_car = grab_col_names(df)

low, high = outlier_thresholds(df, "GENERAL_DAM_RESERVED_WATER")

for col in num_cols:
    print(check_outlier(df, col))

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 12 >= df[col].nunique() > 2]    

df = one_hot_encoder(df, ohe_cols)

# import sklearn.metrics
# def base_models(X, y, scoring="accuracy"):
#     print("Base Models....")
#     classifiers = [('LR', LogisticRegression()),
#                    ('KNN', KNeighborsClassifier()),
#                    ("SVC", SVC()),
#                    ("CART", DecisionTreeClassifier()),
#                    ("RF", RandomForestClassifier()),
#                    ('Adaboost', AdaBoostClassifier()),
#                    ('GBM', GradientBoostingClassifier()),
#                    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='mae')),
#                    ('LightGBM', LGBMClassifier()),
#                    # ('CatBoost', CatBoostClassifier(verbose=False))
#                    ]

#     for name, classifier in classifiers:
#         cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
#         print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

# knn_params = {"n_neighbors": range(1, 50)}

# cart_params = {'max_depth': range(1, 20),
#                "min_samples_split": range(2, 30)}

# rf_params = {"max_depth": [2, 15, None],
#              "max_features": [2, 7, "auto"],
#              "min_samples_split": [2, 20],
#              "n_estimators": [200, 300]}

# xgboost_params = {"learning_rate": [0.1, 0.01],
#                   "max_depth": [5, 8],
#                   "n_estimators": [100, 200],
#                   "colsample_bytree": [0.5, 1]}

# lightgbm_params = {"learning_rate": [0.01, 0.1],
#                    "n_estimators": [300, 500],
#                    "colsample_bytree": [0.7, 1]}

# classifiers = [('KNN', KNeighborsClassifier(), knn_params),
#                ("CART", DecisionTreeClassifier(), cart_params),
#                ("RF", RandomForestClassifier(), rf_params),
#                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
#                ('LightGBM', LGBMClassifier(), lightgbm_params)]

# def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
#     print("Hyperparameter Optimization....")
#     best_models = {}
#     for name, classifier, params in classifiers:
#         print(f"########## {name} ##########")
#         cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
#         print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

#         gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
#         final_model = classifier.set_params(**gs_best.best_params_)

#         cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
#         print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
#         print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
#         best_models[name] = final_model
#     return best_models

# # Stacking & Ensemble Learning
# def voting_classifier(best_models, X, y):
#     print("Voting Classifier...")
#     voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]), ('RF', best_models["RF"]),
#                                               ('LightGBM', best_models["LightGBM"])],
#                                   voting='soft').fit(X, y)
#     cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
#     print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
#     print(f"F1Score: {cv_results['test_f1'].mean()}")
#     print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
#     return voting_clf

# df.columns

# def main():
#     y = df["GENERAL_DAM_RESERVED_WATER"]
#     X = df.drop(["GENERAL_DAM_RESERVED_WATER", "GENERAL_DAM_OCCUPANCY_RATE"], axis=1)
#     base_models(X, y)
#     best_models = hyperparameter_optimization(X, y)
#     voting_clf = voting_classifier(best_models, X, y)
#     joblib.dump(voting_clf, "voting_clf.pkl")
#     return voting_clf

# if __name__ == "__main__":
#     print("İşlem başladı")
#     main()

from sklearn.model_selection import train_test_split



df.drop(labels=["GENERAL_DAM_OCCUPANCY_RATE"], axis = 1, inplace = True)

X_train,X_test, y_train, y_test = train_test_split(df.iloc[:,1:],df.iloc[:,0],test_size=0.3, random_state=17)

import lightgbm as lgb

model = lgb.LGBMRegressor(n_jobs = -1)

model.fit(X_train, y_train)

ypred = model.predict(X_test)

score = model.score(X_test, y_test)

print(score)

params = {'max_depth': range(10,20),
          'colsample_bytree' : [0.5, 0.7, 1],
          'learning_rate': [0.1, 0.01, 0.001],
          'min_child_samples': range(15,25),
          'min_child_weight': [0.001, 0.0001],
          'n_estimators': [50,100,300],
          
          }

model.get_params()

gs_best = GridSearchCV(model, params, cv = None, n_jobs=-1, verbose=True).fit(X_train, y_train)

final_model = model.set_params(**gs_best.best_params_)

final_model = final_model.fit(X_train, y_train)

ypred_final = final_model.predict(X_test)

score_final = final_model.score(X_test, y_test)

print("First Score : " , score , "Final Score : " , score_final)

def plot_importance(model, features, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:-1])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(model, X_train)

import seaborn as sns

sns.lineplot(x = np.linspace(1,100,100), y =  ypred_final[:100])

sns.lineplot(x = np.linspace(1,100,100), y = y_test[:100])


model.get_params()


