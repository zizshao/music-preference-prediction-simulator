import numpy as np
import pandas as pd

def one_hot_enc(df, uniq_feat_list, feat_name):
    """take a list of unique category data to get the one-hot encoding columns"""
    enc = []
    for f in uniq_feat_list:
        enc.append([])
    for f in df[feat_name]:
        for i in range(len(uniq_feat_list)):
            if f == uniq_feat_list[i]:
                enc[i].append(1)
            else:
                enc[i].append(0)
    return enc
    

def train_test_split_by_one_cat(df, col):
    """randomly split the training and testing set according
       to one category (the proportion of each category is
       preserved across the training and testing dataset)"""
    train = []
    test = []
    unq_cat = sorted(df[col].unique())
    for c in unq_cat:
        df_onlyc = df[df[col] == c].sample(frac=1, random_state=42)
        train_size = int(0.8 * len(df_onlyc))
        test_size = len(df_onlyc) - train_size
        train.append(df_onlyc[:train_size])
        test.append(df_onlyc[train_size:])
    train_df = pd.concat(train, ignore_index=True)
    test_df = pd.concat(test, ignore_index=True)
    return (train_df, test_df)

def stz(df, df_tr, stz_features):
    """standardize the features based on the mean of df_tr"""
    df_stz = df.copy()
    for sf in stz_features:
        m = df_tr[sf].mean()
        std = df_tr[sf].std()
        df_stz[sf] = (df_stz[sf] - m)/std
    return df_stz

def MSE(y, y_hat):
    """mean squared error"""
    se = np.sum((y - y_hat)**2)
    return se/y.shape[0]

def get_class_mean(traindf, class_column_name, features):
    cls_mean = []
    for c in traindf[class_column_name].unique():
        m = traindf[traindf[class_column_name]==c][features].mean().to_numpy()
        cls_mean.append((m,c))
    return cls_mean

def min_dist_classify(testdf, cls_mean_list):
    """return the result of classification by using minimum distance"""
    pred_cls = []
    for row in testdf:
        dist_all = []
        for m,c in cls_mean_list:
            dist = np.linalg.norm(row - m)
            dist_all.append((dist,c))
        min_dist, pred = min(dist_all)
        pred_cls.append(pred)
    return np.array(pred_cls)
        
