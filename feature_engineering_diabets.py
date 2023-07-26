#####################################################################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING PROJECT (DIABETS)
#####################################################################################
#####################################
# Gerekli kütüphanelerin yüklenmesi
# Import libraries we need
#####################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

################################
# Genel Bakış
# Overview
################################
df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape
df.info()
df.isnull().sum()
#################################
# num_cols ve cat_cols'ların yakalanması
# Catching num_cols and cat_cols
#################################
for col in df.columns:
    print(col, df[col].nunique())

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

#################################
# Hedef - Değişken analizi
# Target - Variable analysis
#################################

def target_summary_with_cat(dataframe, target,caterogical_col):
    print(dataframe.groupby(caterogical_col)[target].mean())

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}))

for cat in cat_cols:
    target_summary_with_cat(df, "Outcome", cat)

for num in num_cols:
    target_summary_with_num(df, "Outcome", num)

#################################
# Aykırı Değerlerin Analizi
# Outliers Analysis
#################################

def outlier_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col,check_outlier(df, col))

#################################
# Eksik Değer Analizi
# Missing Values Analysis
#################################

df.isnull().sum()

df[(df["Glucose"] == 0) | df["Insulin"] == 0].shape

df['Glucose'] = df['Glucose'].replace(0, float('nan'))

df['Insulin'] = df['Insulin'].replace(0, float('nan'))

####################################
# Korelasyon Analizi
# Analysis of Corelatıon
####################################

corr = df[num_cols].corr()
print(corr)

for col in corr:
    corr[col] = corr[col].replace(1, 0)

#####################################
# Eksik ve Aykırı Değerlerin Düzenlenmesi
# Editing Missing and Outlier Values
#####################################

# Outliers
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    #dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

# Missing Values
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

scaler = MinMaxScaler()

dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)

dff.head()

# KNN
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df['Glucose_imputed_knn'] = dff["Glucose"]
df["Insulin_imputed_knn"] = dff["Insulin"]

df.loc[df["Insulin"].isnull(),["Insulin", "Insulin_imputed_knn"]]
df.loc[df["Insulin"].isnull()]

df.loc[df["Glucose"].isnull(),["Glucose", "Glucose_imputed_knn"]]
df.loc[df["Glucose"].isnull()]

df[["Insulin", "Insulin_imputed_knn"]].head()

df["Insulin"] = df["Insulin_imputed_knn"]
df["Glucose"] = df["Glucose_imputed_knn"]

df.drop("Insulin_imputed_knn", axis=1, inplace=True)
df.drop("Glucose_imputed_knn", axis=1, inplace=True)

df.isnull().sum()

#####################################
# Yeni Değişkenlerin Oluşturulması
# Creating new Features
#####################################
df.head()

df.describe().T

df.groupby("Outcome")["DiabetesPedigreeFunction"].mean()

df.loc[df["Pregnancies"] < 4, "Pregnancies_new"] = "low"

df.loc[df["Pregnancies"] > 4, "Pregnancies_new"] = "high"

df.loc[df["BMI"] <= 18.5, "BMI_CAT"] = "weak"

df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 25), "BMI_CAT"] = "normal"

df.loc[(df["BMI"] > 25) & (df["BMI"] <= 30), "BMI_CAT"] = "fat"

df.loc[(df["BMI"] > 30) & (df["BMI"] <= 35), "BMI_CAT"] = "obesity_low"

df.loc[df["BMI"] > 35, "BMI_CAT"] = "obesity_high"

df["Glucose_CAT"] = pd.qcut(df["Glucose"], 5, ["very_low","low", "normal", "high", "very_high"])

df["Insulin_CAT"] = pd.qcut(df["Insulin"], 5, ["very_low","low", "normal", "high", "very_high"])

df["BloodPresure_CAT"] = pd.qcut(df["BloodPressure"], 5, ["very_low","low", "normal", "high", "very_high"])

df["Age_X_DiabetesPedigreeFunction"] = df["Age"] * df["DiabetesPedigreeFunction"]

df["Age_X_BloodPressure"] = df["Age"] * df["BloodPressure"]

#
df["Pregnancies_x_DPF"] = df["DiabetesPedigreeFunction"] * df["Pregnancies"]

df["SkinThickness_x_DPF"] = df["SkinThickness"] * df["DiabetesPedigreeFunction"] / 100

df["Blood_pressure_x_DPF"] = df["DiabetesPedigreeFunction"] / df["BloodPressure"]

df["BMI_x_DPF"] = df["DiabetesPedigreeFunction"] / df["BMI"]

df["Age_x_BMI"] = df["Age"] / df["BMI"]

df["Age_x_DPF"] = df["Age"] * df["DiabetesPedigreeFunction"] / 100

df["Age_x_Blood_pressure"] = df["Age"] * df["BloodPressure"] / 100

df["BMI_x_Blood_pressure"] = df["BMI"] * df["BloodPressure"] / 100

df["Glucose_x_Insulin"] = df["Glucose"] / df["Insulin"]

df["Insulin_x_DPF"] = df["Insulin"] * df["DiabetesPedigreeFunction"]

df["Insulin_y_DPF"] = df["Insulin"] / df["DiabetesPedigreeFunction"]

df["Glucose_x_DPF"] = df["Glucose"] * df["DiabetesPedigreeFunction"]

df["Glucose_y_DPF"] = df["Glucose"] / df["DiabetesPedigreeFunction"]

df["Glucose_Insulin_x_DPF"] = df["Glucose_x_Insulin"] * df["DiabetesPedigreeFunction"]

df["Skin_Thickness_x_Blood_pressure"] = df["SkinThickness"] * df["BloodPressure"] / 100

df["Skin_Thickness_x_Glucose_Insulin"] = df["SkinThickness"] / df["Glucose_x_Insulin"]

df["Skin_Thickness_x_Age"] = df["SkinThickness"] / df["Age"]

df["Skin_Thickness_x_BMI"] = df["SkinThickness"] / df["BMI"]


##################################
# Encoding
##################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)

##### Binary Encoding
binary_cols = [col for col in cat_cols if df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)
###### One-Hot Encoding
ohe_cols = [col for col in cat_cols if col not in binary_cols]

def one_hot_encoder(dataframe, ohe_col, drop_first=True):
    dataframe = pd.get_dummies(dataframe,columns= ohe_col, drop_first= drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols)

df.head()
#############################################
# Standart Scaler
#############################################
scaler = StandardScaler()

df = df.replace([np.inf, -np.inf], np.nan).dropna()


df[num_cols] = scaler.fit_transform(df[num_cols])

##############################################
# Model
##############################################

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = df.drop(columns=['Outcome'])
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion)
print("\nClassification Report:")
print(classification_rep)

df.head()
df.describe().T

#######################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(model, X_train)
