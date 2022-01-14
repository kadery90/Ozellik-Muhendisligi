import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv(r"C:\Users\yildi\OneDrive\Masaüstü\datasets\diabetes.csv")
df_=df.copy()

df.head()
df.shape

##Numerik ve Kategorik Değişkenler

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
        car_th: int, optional
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

##cat_cols ['Outcome']

##num_cols
#['Pregnancies',
 #'Glucose',
 #'BloodPressure',
 #'SkinThickness',
 #'Insulin',
 #'BMI',
 #'DiabetesPedigreeFunction',
 #'Age']

df[num_cols].describe().T

##Hedef Değişken Analizi

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

##Aykırı Gözlem

sns.boxplot(x=df["BloodPressure"])
plt.show()

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

for col in num_cols:
    print(col, check_outlier(df, col))

#Pregnancies True
#Glucose True
#BloodPressure True
#SkinThickness True
#Insulin True
#BMI True
#DiabetesPedigreeFunction True
#Age True

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

##Eksik Gözlem

df.isnull().values.any()  ##False

##Korelasyon Analizi

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

##Feature Engineering

def assing_missing_values(dataframe, except_cols):
    for col in dataframe.columns:
        dataframe[col] = [val if val!=0 or col in except_cols else np.nan for val in df[col].values]
    return dataframe

df = assing_missing_values(df, except_cols=['Outcome', 'Pregnancies'])

df.isnull().values.any()  ##True

df.isnull().sum()

#Glucose                       5
#BloodPressure                35
#SkinThickness               227
#Insulin                     374
#BMI                          11

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)

df.dropna().shape  ##392 silmiyorum.

##eksik değerleri inceleme:

msno.bar(df)
plt.show()

##Eksik değerlerin Outcome ile Analizi

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Outcome", na_cols)

##Eksik Değer Doldurma-KNNile

cat_cols, num_cols, cat_but_car = grab_col_names(df)
df = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

df.head()

df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)

df.head()

##Aykırı Değer

for col in num_cols:
    print(col, check_outlier(df, col))

#Pregnancies True
#Glucose False
#BloodPressure True
#SkinThickness True
#Insulin True
#BMI True
#DiabetesPedigreeFunction True
#Age True

##Baskılama

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#Pregnancies False
#Glucose False
#BloodPressure False
#SkinThickness False
#Insulin False
#BMI False
#DiabetesPedigreeFunction False
#Age False

##Yeni Değişken Türetme

df.loc[(df['Age'] < 18), 'New_Age_Cat'] = 'ergen'
df.loc[(df['Age'] >= 18) & (df['Age'] < 24), 'New_Age_Cat'] = 'genc'
df.loc[(df['Age'] >= 24) & (df['Age'] < 35), 'New_Age_Cat'] = 'yetiskin'
df.loc[(df['Age'] >= 35) & (df['Age'] < 45), 'New_Age_Cat'] = 'gencortayas'
df.loc[(df['Age'] >= 45) & (df['Age'] < 55), 'New_Age_Cat'] = 'ortayas'
df.loc[(df['Age'] >= 55), 'New_Age_Cat'] = 'yasli'

df.head()

df.loc[(df['BMI'] < 18.5), 'NEW_BMI'] = 'zayif'
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 25), 'NEW_BMI'] = 'normal'
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30), 'NEW_BMI'] = 'fazlakilolu'
df.loc[(df['BMI'] >= 30) & (df['BMI'] < 45), 'NEW_BMI'] = 'obez'
df.loc[(df['BMI'] >= 45), 'NEW_BMI'] = 'asiriobez'

df.head()

df.loc[(df['BloodPressure'] < 60), 'New_BP'] = 'düsüktansiyon'
df.loc[(df['BloodPressure'] >= 60) & (df['BloodPressure'] < 80), 'New_BP'] = 'idealtansiyon'
df.loc[(df['BloodPressure'] >= 80) & (df['BloodPressure'] < 90), 'New_BP'] = 'sagliklitansiyon'
df.loc[(df['BloodPressure'] >= 90), 'New_BP'] = 'yüksektansiyon'

df.head()

##Encoding

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

##binarycols yoktur.

## rare

def rare_encoder(dataframe, rare_perc, cat_cols):
    # 1'den fazla rare varsa düzeltme yap. durumu göz önünde bulunduruldu.
    # rare sınıf sorgusu 0.01'e göre yapıldıktan sonra gelen true'ların sum'ı alınıyor.
    # eğer 1'den büyük ise rare cols listesine alınıyor.
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col])

    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "Outcome", cat_cols)

rare_encoder(df, 0.01, cat_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df = rare_encoder(df, 0.01, cat_cols)

##onehot

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

## ['New_Age_Cat', 'NEW_BMI', 'New_BP']

df = one_hot_encoder(df, ohe_cols)

df.head()

df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)
rare_analyser(df, "Outcome", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

useless_cols  ##['NEW_BMI_zayif']

df.drop(useless_cols, axis=1, inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

##standartlaştırma

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

##model

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

##0.8138528138528138

##değişkenlerin önemi

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


plot_importance(rf_model, X_train, num=30)



