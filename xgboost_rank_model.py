# -*-- coding:utf-8 -*--
import sys
import xgboost as xgb
import pandas as pd

def rank_run(dtrain_path, dtrain_g_path, dval_path, dval_g_path, dtest_path, dtest_g_path):
    params = {"objective": "rank:pairwise",
              "eta": 0.1,
              "gamma": 1.0,
              "min_child_weight": 1,
              "max_depth": 6}

    dtrain = xgb.DMatrix(dtrain_path)
    dtrain_group = xgb.DMatrix(dtrain_g_path)
    dtrain.set_group(dtrain_group)
    
    dval = xgb.DMatrix(dval_path)
    dval_group = xgb.DMatrix(dval_g_path)
    dval.set_group(dval_group) 


    dtest = xgb.DMatrix(dtest_path)
    dtest_group = xgb.DMatrix(dtest_g_path)
    dtest.set_group(dtest_group)

    ## 需要准备一个watchlist 给train 和 validation
    watchlist = [(dval, 'eval'), (dtrain, 'train')]
    num_round = 4
    bst = xgb.train(params, dtrain, num_round, wathchlist)

    predict = bst.predict(dtest)
    
    print ("done....")

def load_group_file(path):
    group = []

    with open(path, 'r') as file:
        for line in file:
            try:
                group.append(int(line.strip()))

            except Exception as ex:
                print "Exception happen at line:", line
    return group

