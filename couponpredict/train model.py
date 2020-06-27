import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# 读取数据
def get_processed_data():
    dataset1 = pd.read_csv('myspace/ProcessDataSet1.csv')
    dataset2 = pd.read_csv('myspace/ProcessDataSet2.csv')
    dataset3 = pd.read_csv('myspace/ProcessDataSet3.csv')

    dataset1.drop_duplicates(inplace=True)
    dataset2.drop_duplicates(inplace=True)
    dataset3.drop_duplicates(inplace=True)

    dataset12 = pd.concat([dataset1, dataset2], axis=0)

    dataset12.fillna(0, inplace=True)
    dataset3.fillna(0, inplace=True)

    return dataset12, dataset3


dataset12, dataset3 = get_processed_data()

dataset12_x = dataset12.drop(
    columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
             'Date', 'Coupon_id', 'label'], axis=1)
dataset3_x = dataset3.drop(
    columns=['User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
             'Coupon_id'], axis=1)

categorical_features_indices = np.where(dataset12_x.dtypes != np.float)[0]


model = CatBoostClassifier(iterations=100, depth=5,cat_features=categorical_features_indices,learning_rate=0.5, loss_function='Logloss',
                            logging_level='Verbose')

dataset12_y  = dataset12["label"]

model.fit(dataset12_x,dataset12_y,plot=True)
# 可视化特征重要性
import matplotlib.pyplot as plt
fea_ = model.feature_importances_
fea_name = model.feature_names_
plt.figure(figsize=(20, 20))
plt.barh(fea_name,fea_,height =0.5)

preds_class = model.predict(dataset3_x)

preds_proba = model.predict_proba(dataset3_x)

dftest1 = dataset3[['User_id','Coupon_id','Date_received']].copy()
dftest1['label'] = preds_proba[:,1]
dftest1.to_csv('submit1.csv', index=False, header=False)
dftest1.head()