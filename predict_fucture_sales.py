# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:56:54 2020

@author: Doubleshun
"""

# Goal : predict total sales for every product and store in the next month.

############################################################################################################
############################################ File descriptions #############################################
# sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.             #
# test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.  #
# sample_submission.csv - a sample submission file in the correct format.                                  #
# items.csv - supplemental information about the items/products.                                           #
# item_categories.csv  - supplemental information about the items categories.                              #
# shops.csv- supplemental information about the shops.                                                     #
############################################################################################################
import pandas as pd
import seaborn as sns
import numpy as np
import xgboost as xgb
import datetime
import itertools
from tqdm import tqdm

'''
STEP 1 : 匯入資料
''' 
item_categories = pd.read_csv(r'E:\400_PersonalStudy\python\Predict_Future_Sales\0_Dataset\item_categories.csv')
items = pd.read_csv(r'E:\400_PersonalStudy\python\Predict_Future_Sales\0_Dataset\items.csv')
sales_train = pd.read_csv(r'E:\400_PersonalStudy\python\Predict_Future_Sales\0_Dataset\sales_train.csv')
sample_submission = pd.read_csv(r'E:\400_PersonalStudy\python\Predict_Future_Sales\0_Dataset\sample_submission.csv')
shops = pd.read_csv(r'E:\400_PersonalStudy\python\Predict_Future_Sales\0_Dataset\shops.csv')
test = pd.read_csv(r'E:\400_PersonalStudy\python\Predict_Future_Sales\0_Dataset\test.csv')
'''
STEP 2 : 資料前處理
'''
#(1)資料格式轉換 - 將「sales_train」資料集的日期格式從「日－月－年」改成「年－月－日」
sales_train['date'] = pd.to_datetime(sales_train.date,format="%d.%m.%Y")

#(2)刪除離群值 - 畫箱型圖判斷離群值
sns.set_style("whitegrid")
sns.boxplot(x=sales_train.item_cnt_day)
sales_train_clean = sales_train[sales_train['item_cnt_day']<1000]
sns.boxplot(x=sales_train.item_price)
sales_train_clean = sales_train[sales_train['item_price']<100000]

#(3)檢查合理性
item_price = pd.DataFrame(set(sales_train.item_price),columns=['item_price']).sort_values(by=['item_price']) #價格中有負的
sales_train_clean = sales_train[sales_train['item_price']>0]
item_cnt_day = pd.DataFrame(set(sales_train.item_cnt_day),columns=['item_cnt_day']).sort_values(by=['item_cnt_day']) #購買數量有負的
sales_train_clean = sales_train[sales_train['item_cnt_day']>0]

#(4)將日銷量改為月銷量
'''
數據主要有三個特徵：shop_id, item_id, item_cnt_day
'''
train_table = pd.pivot_table(sales_train_clean, index=['shop_id','item_id','date_block_num'], values=['item_cnt_day'], aggfunc=[np.sum],fill_value=0).reset_index()
train_table.columns = train_table.columns.droplevel().map(str)
train_table.columns.values[0]="shop_id"
train_table.columns.values[1]="item_id"
train_table.columns.values[2]="month_id"
train_table.columns.values[3]="item_cnt_month"

#(5)補上缺值
'''
數據只包含了商店真正銷售過的商品每月銷量
但是如果商店該月没有銷售該商品，則上面的數據中則沒有該條數據
而實際上這時候該商店該月該商品的銷售量應該等於0
'''
month_list = list(set(train_table.month_id))
shop_list = list(set(train_table.shop_id))
item_list = list(set(train_table.item_id))
shop_in_all_month = pd.DataFrame(list(itertools.product(shop_list,month_list)),columns=['shop_id','month_id'])

train_table_after = train_table
for idx,row in tqdm(shop_in_all_month.iterrows()):
    print(str(idx)+' / '+str(len(shop_in_all_month)))
    tmp_shop_month_sales = train_table[( train_table.shop_id == row['shop_id'] ) & ( train_table.month_id == row['month_id'] )]
    item_list = set(train_table.item_id)
    tmp_shop_month_sales_item_list = set(tmp_shop_month_sales.item_id)
    diff = item_list - tmp_shop_month_sales_item_list
    tmp_shop_month_sales_lost = pd.DataFrame(columns=train_table.columns)
    tmp_shop_month_sales_lost['item_id'] = list(diff)
    tmp_shop_month_sales_lost['shop_id'] = tmp_shop_month_sales_lost['shop_id'].fillna( row['shop_id'] )
    tmp_shop_month_sales_lost['month_id'] = tmp_shop_month_sales_lost['month_id'].fillna( row['month_id'] )
    tmp_shop_month_sales_lost['item_cnt_month'] = tmp_shop_month_sales_lost['item_cnt_month'].fillna(0)
    train_table_after = train_table_after.append(tmp_shop_month_sales_lost)
    
#(6)增加特徵
'''
特徵1:該商品上個月在所有商店的總銷量
特徵2:該商店上個月的總銷售量
特徵3:商品上個月在不同商店的平均價格
特徵4:商品類型
'''
###特徵1:該商品上個月在所有商店的總銷量
item_all_sales_lastM = pd.pivot_table(train_table_after, index=['item_id','month_id'], values=['item_cnt_month'], aggfunc=[np.sum],fill_value=0).reset_index()
item_all_sales_lastM.columns = item_all_sales_lastM.columns.droplevel().map(str)
item_all_sales_lastM.columns.values[0]="item_id"
item_all_sales_lastM.columns.values[1]="month_id"
item_all_sales_lastM.columns.values[2]="item_cnt_month"
item_all_sales_lastM["month_id"] = item_all_sales_lastM["month_id"] + 1 #加1後，merge才會merge到下個月
###特徵2:該商店上個月的總銷售量
shop_all_sales_lastM = pd.pivot_table(train_table_after, index=['shop_id','month_id'], values=['item_cnt_month'], aggfunc=[np.sum],fill_value=0).reset_index()
shop_all_sales_lastM.columns = shop_all_sales_lastM.columns.droplevel().map(str)
shop_all_sales_lastM.columns.values[0]="shop_id"
shop_all_sales_lastM.columns.values[1]="month_id"
shop_all_sales_lastM.columns.values[2]="shop_cnt_month_lastM"
shop_all_sales_lastM["month_id"] = shop_all_sales_lastM["month_id"] + 1
###特徵3:商品上個月在不同商店的平均價格
item_price_mean = pd.pivot_table(sales_train_clean, index=['item_id','date_block_num'], values=['item_price'], aggfunc=[np.mean],fill_value=0).reset_index()
item_price_mean.columns = item_price_mean.columns.droplevel().map(str)
item_price_mean.columns.values[0]="item_id"
item_price_mean.columns.values[1]="month_id"
item_price_mean.columns.values[2]="item_price_mean_lastM"
item_price_mean["month_id"] = item_price_mean["month_id"] + 1
###特徵4:商品類型
item_id_categories = items[['item_id','item_category_id']]
###結合test，並將上面新加入之特徵Merge回train_table_after
test_data = test[['item_id','shop_id']]
test_data['month_id']=34
combined_data = train_table_after.append(test_data)
combined_data = pd.merge(combined_data,item_all_sales_lastM,on=['item_id','month_id'],how='left')
combined_data = pd.merge(combined_data,shop_all_sales_lastM,on=['shop_id','month_id'],how='left')
combined_data = pd.merge(combined_data,item_price_mean,on=['item_id','month_id'],how='left')
combined_data = pd.merge(combined_data,item_id_categories,on='item_id',how='left')
x_train = combined_data[combined_data['month_id']<34].drop(['item_cnt_month'],axis=1)
x_test = combined_data[combined_data['month_id']==34].drop(['item_cnt_month'],axis=1).fillna(0)
y_train = combined_data[combined_data['month_id']<34][['item_cnt_month']]
'''
STEP 3 : 建立模型(XGBOOST)與預測
'''
model = xgb.XGBRegressor(max_depth=4, colsample_btree=0.1, learning_rate=0.1, n_estimators=32, min_child_weight=2);
model.fit(x_train, y_train,eval_metric = 'rmse')
y_test = model.predict(x_test)

submission = pd.DataFrame(y_test,columns=['item_cnt_month'])
submission.index.name = 'ID'
submission.to_csv(r'E:\400_PersonalStudy\python\Predict_Future_Sales\2_SubmitFiles\predict_future_sales_%s.csv'%datetime.date.today())
