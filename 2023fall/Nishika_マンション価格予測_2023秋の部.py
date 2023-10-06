#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import os
import warnings 
warnings.simplefilter('ignore')

import mojimoji
import lightgbm as lgb
#import optuna.integration import lightgbm as lgb
import unicodedata
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.model_selection import KFold


print("process has started")


temp_input = []

for filename in os.listdir("train/train/"):
    temp_input.append(pd.read_csv("train/train/"+filename))

df = pd.concat(temp_input, axis=0, ignore_index=True)
pd.set_option('display.max_columns', None)

# データが完全に欠損している列は最初に削除
df.drop(columns=["地域","土地の形状","間口","延床面積（㎡）","前面道路：方位","前面道路：種類","前面道路：幅員（ｍ）"],inplace=True)


#print(df["種類"].value_counts()) # 全て同じ値であるため、使い道は無いので削除
df.drop(columns="種類",inplace=True)


df.replace({"最寄駅：距離（分）":{"30分?60分":"30","1H?1H30":"60","1H30?2H":"90","2H?":"120"}},inplace=True)
df["最寄駅：距離（分）"]=pd.to_numeric(df["最寄駅：距離（分）"], errors='coerce') 
df.replace({"面積（㎡）":{"2000㎡以上":"2000"}},inplace=True)
df["面積（㎡）"]=pd.to_numeric(df["面積（㎡）"], errors='coerce') 

years = []
for y in  df["建築年"]:
    tmp = re.sub("\\D", "", str(y))
    if "元年" in str(y) or (tmp.isdigit()):
        if "平成" in y:
            if y=="平成元年":
                year = 1989
            else:
                year = 1989 + int(re.sub("\\D", "", y)) - 1
        elif "昭和" in str(y):
            if y=="昭和元年":
                year = 1926
            else:
                year = 1926 + int(re.sub("\\D", "", y)) - 1
        elif "令和" in str(y):
            if y=="令和元年":
                year = 2019
            else:
                year = 2019 + int(re.sub("\\D", "", y)) - 1
    elif "戦前" in str(y):
        year=1940
    else:
        year=y
    years.append(year)
df["建築西暦年"] =pd.Series(years)

df["取引時点"].value_counts() # →　取引年と四半期に分解 →　取引年と建築年から築年数を算出、四半期は傾向を別途調査
df["取引年"]=df["取引時点"].apply(lambda x :  int(x[0:4]))
df["取引四半期"]=df["取引時点"].apply(lambda x :  int(unicodedata.normalize('NFKC', x[6:7])))
df["取引年"] = df["取引年"] + df["取引四半期"]*0.25 -0.125


durations = []
for i in range(len(df)):
    if df["取引年"].iloc[i] > 0 and df["建築西暦年"].iloc[i] > 0:
        duration = df["取引年"].iloc[i] - df["建築西暦年"].iloc[i]
    else:
        duration=None
    durations.append(duration)

    
df["購入までの築年数"] = pd.Series(durations)


df["部屋数"] = df["間取り"].str[0].map(lambda x: int(mojimoji.zen_to_han(str(x)) ) if mojimoji.zen_to_han(str(x)).isdigit() else 0)

df_X = df.drop(columns=["ID","取引価格（総額）_log","建築年","取引時点"]) # 
df_y = df["取引価格（総額）_log"]

print("creating training data has finished")

categorical_feature = [
    #"市区町村コード",
    "都道府県名",
    "市区町村名",
    "地区名",
    "最寄駅：名称",
    "間取り",
    "建物の構造",
    "用途",
    "今後の利用目的",
    "都市計画",
    "改装",
    "取引の事情等",
    "建築西暦年",
    #"取引西暦年",
    #"取引時点",
    #"取引四半期",
    #"建築年",
    #"取引時点"
    ]

for c in categorical_feature:
    df_X[c] = df_X[c].astype("category")



train_X,eval_X,train_y,eval_y = train_test_split(df_X,df_y,random_state=12345,test_size=0.05,stratify=df["都道府県名"])

lgb_train = lgb.Dataset(train_X, train_y,free_raw_data=False)
lgb_eval  = lgb.Dataset(eval_X, eval_y, reference=lgb_train,free_raw_data=False)

params = {
          'task': 'train',              # タスクを訓練に設定
          'boosting_type': 'gbdt', # GBDT     
          'objective': 'regression_l1',    # 回帰を指定 L1
          'metric': 'mae',             # 回帰の評価関数
          'learning_rate': 0.01,         # 学習率
          'num_leaves':1000,
          'max_depth': -1,
          'reg_alpha':2,
          'reg_lambda':2,
          'colsample_bytree':0.5,
          'subsample':0.5,
          'subsample_freq':1,
          'min_child_samples':20,
          #'tree_type':'data',
          #'data_sample_strategy':'goss',
          #'device':'gpu'
          }

model = lgb.train(
                  params=params,                    # ハイパーパラメータをセット
                  train_set=lgb_train,              # 訓練データを訓練用にセット
                  valid_sets=[lgb_train, lgb_eval], # 訓練データとテストデータをセット
                  valid_names=['Train', 'Eval'],    # データセットの名前をそれぞれ設定
                  num_boost_round=50000,             # 計算回数
                  categorical_feature = categorical_feature,
                  keep_training_booster=True,
                  callbacks=[lgb.early_stopping(stopping_rounds=100, 
                             verbose=True), # early_stopping用コールバック関数
                             lgb.log_evaluation(100)] # コマンドライン出力用コールバック関数
                
                  )  
print(model.params)

df_test = pd.read_csv("test.csv")

df_test.drop(columns=["地域","土地の形状","間口","延床面積（㎡）","前面道路：方位","前面道路：種類","前面道路：幅員（ｍ）"],inplace=True)
df_test.drop(columns="種類",inplace=True)
# データが完全に欠損している列は最初に削除

df_test.replace({"最寄駅：距離（分）":{"30分?60分":"30","1H?1H30":"60","1H30?2H":"90","2H?":"120"}},inplace=True)
df_test["最寄駅：距離（分）"]=pd.to_numeric(df_test["最寄駅：距離（分）"], errors='coerce') 

df_test.replace({"面積（㎡）":{"2000㎡以上":"2000"}},inplace=True)

df_test["面積（㎡）"]=pd.to_numeric(df_test["面積（㎡）"], errors='coerce') 

years_test = []
for y in  df_test["建築年"]:
    #print(y)
    tmp = re.sub("\\D", "", str(y))
    if "元年" in str(y) or (tmp.isdigit()):
        if "平成" in y:
            if y=="平成元年":
                year = 1989
            else:
                year = 1989 + int(re.sub("\\D", "", y)) - 1
        elif "昭和" in str(y):
            if y=="昭和元年":
                year = 1926
            else:
                year = 1926 + int(re.sub("\\D", "", y)) - 1
        elif "令和" in str(y):
            if y=="令和元年":
                year = 2019
            else:
                year = 2019 + int(re.sub("\\D", "", y)) - 1
    elif "戦前" in str(y):
        year=1940
    else:
        year=y
    years_test.append(year)
df_test["建築西暦年"] =pd.Series(years_test)

df_test["取引時点"].value_counts() # →　取引年と四半期に分解 →　取引年と建築年から築年数を算出、四半期は傾向を別途調査
df_test["取引年"]=df_test["取引時点"].apply(lambda x :  int(x[0:4]))
df_test["取引四半期"]=df_test["取引時点"].apply(lambda x :  int(unicodedata.normalize('NFKC', x[6:7])))
df_test["取引年"] = df_test["取引年"] + df_test["取引四半期"]*0.25 -0.125

durations_test = []
for i in range(len(df_test)):
    if df_test["取引年"].iloc[i] > 0 and df_test["建築西暦年"].iloc[i] > 0:
        duration = df_test["取引年"].iloc[i] - df_test["建築西暦年"].iloc[i]
    else:
        duration=None
    durations_test.append(duration)
df_test["購入までの築年数"] = pd.Series(durations_test)
df_test["部屋数"] = df_test["間取り"].str[0].map(lambda x: int(mojimoji.zen_to_han(str(x)) ) if mojimoji.zen_to_han(str(x)).isdigit() else 0)
df_test_X = df_test.drop(columns=["ID","建築年","取引時点"]) #,"建築年","取引時点"

for c in categorical_feature:
    df_test_X[c]=df_test_X[c].astype("category")
print(df_test_X.info())
print("creating test data has finished")

pred_y = model.predict(df_test_X) # predict_disable_shape_cheke=Trueを使うと深刻な計算間違いをする
df_pred = pd.DataFrame([],columns=["ID","取引価格（総額）_log"])
df_pred["取引価格（総額）_log"] = pd.Series(pred_y)
df_pred['ID'] = df_test["ID"]

print(df_pred)

df_pred.to_csv("walker_mansion_2023fall.csv",index=False)


