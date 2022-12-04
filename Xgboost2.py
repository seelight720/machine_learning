from sklearn import datasets
# 利用model_selection进行交叉训练
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from xgboost import plot_importance  # 显示特征重要性
import numpy as np
from xgboost import XGBRegressor


def loaddata(txt_path, delimiter):
    # ---
    # 功能：读取只包含数字的txt文件，并转化为array形式
    # txt_path：txt的路径；delimiter：数据之间的分隔符
    # ---
    data_list = []
    label_list=[]
    with open(txt_path) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")  # 去除末尾的换行符
        data_split = line.split(delimiter)
        temp_data = list(map(float, data_split[0:-8]))
        #temp_data = list(map(float, data_split[0:-1]))
        data_list.append(temp_data)

        temp_label=list(map(float, data_split[-8:]))

        #label_list.append(float(data_split[-1]))
        label_list.append(temp_label)

    data_array = np.array(data_list)
    label_array = np.array(label_list)
    return data_array,label_array


def xgboost(data_array,label_array):
    iris = datasets.load_iris()

    param = {'max_depth': 2, 'eta': 1, 'silent': 0, 'objective': 'binary:logistic'}
    num_round = 100
    bst2 = XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=num_round, silent=True, objective='binary:logistic')
    # bst2.fit(iris.data, iris.target)
    bst2.fit(data_array, label_array)
    kfold = StratifiedKFold(n_splits=10, random_state=7,shuffle=True)
    # results = cross_val_score(bst2, iris.data, iris.target, cv=kfold)  # 对数据进行十折交叉验证--9份训练，一份测试
    results = cross_val_score(bst2,data_array, label_array, cv=kfold)  # 对数据进行十折交叉验证--9份训练，一份测试

    print(results)
    print("CV Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    plot_importance(bst2)  # 打印重要程度结果。
    pyplot.show()

def xgboost_regressor(data_array, label_array):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from matplotlib import pyplot as plt
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler  # 最大最小归一化
    from sklearn.preprocessing import StandardScaler  # 标准化
    # label_encoder = LabelEncoder()
    # y = label_encoder.fit_transform(label_array)
    X_train, X_test, y_train, y_test = train_test_split(data_array, label_array, test_size=0.25, random_state=7)
    X_train=pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    ss = MinMaxScaler()
    #ss=StandardScaler()
    X_train2 = ss.fit_transform(X_train)
    X_test2= ss.fit_transform(X_test)

    #model = XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=100, silent=True)
    model = XGBRegressor(subsample=0.9, reg_lambda=0.008, reg_alpha=0.1, n_estimators=61, min_child_weight=5.49, max_depth=6, learning_rate=0.15, gamma=0.0, colsample_bytree=0.7, silent=True)
    model.fit(X_train2, y_train)
    y_test_pre = model.predict(X_test2)
    rmse = sqrt(mean_squared_error(np.array(list(y_test)), np.array(list(y_test_pre))))
    print("rmse:", rmse)
    plot_importance(model)
    plt.show()


    # # # 2.调参方法1
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    param_grid = {
        'max_depth': [2, 3, 4, 5, 6, 7, 8],
        'n_estimators': [30, 50, 100, 300, 500, 1000, 2000],
        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.03, 0.05, 0.5],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "reg_alpha": [0.0001, 0.001, 0.01, 0.1, 1, 100],
        "reg_lambda": [0.0001, 0.001, 0.01, 0.1, 1, 100],
        "min_child_weight": [2, 3, 4, 5, 6, 7, 8],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "subsample": [0.6, 0.7, 0.8, 0.9]}
    # 3.随机搜索并打印最佳参数
    from sklearn.model_selection import RandomizedSearchCV
    gsearch1 = RandomizedSearchCV(XGBRegressor(scoring='ls', seed=27), param_grid, cv=5)
    gsearch1.fit(X_train2, y_train)
    print("best_score_:", gsearch1.best_params_, gsearch1.best_score_)
    # 4.用最佳参数进行预测
    y_test_pre = gsearch1.predict(X_test2)

    # 5.打印测试集RMSE
    rmse = sqrt(mean_squared_error(np.array(list(y_test)), np.array(list(y_test_pre))))
    print("rmse:", rmse)
    # #--调参方法2
    # ## 需要调的参数
    # cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
    # ## 其他基本参数
    # other_params = {'learning_rate': 0.1,
    #                 'n_estimators': 500,
    #                 'max_depth': 5,
    #                 'min_child_weight': 1,
    #                 'seed': 0,
    #                 'subsample': 0.8,
    #                 'colsample_bytree': 0.8,
    #                 'gamma': 0, 'reg_alpha': 0,
    #                 'reg_lambda': 1}



if __name__ == '__main__':
    #data_array,label_array=loaddata("testdata.txt",",")
    #data_array, label_array = loaddata("类别0时间多维.txt", ",")
    #data_array, label_array = loaddata("类别0.txt", ",")
    #data_array, label_array = loaddata("类别3.txt", ",")
    #data_array, label_array = loaddata("类别1.txt", ",")
    #data_array, label_array = loaddata("类别1时间1维.txt", ",")
    #data_array, label_array = loaddata("类别2.txt", ",")
    data_array, label_array = loaddata("类别5.txt", ",")
    #data_array, label_array = loaddata("类别4.txt", ",")
    #data_array, label_array = loaddata("类别4不把路网密度放第一列.txt", ",")
    # xgboost(data_array,label_array)
    xgboost_regressor(data_array, label_array)