from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import preprocessing
# 加载数据集
#dataset = loadtxt('E:/file/pima-indians-diabetes.csv', delimiter=",")
dataset=pd.read_excel('F:/迁居文献/10-28语义识别结果/XGBOOST/类别01.xlsx')
'''
# 将数据分为 数据和标签
'''
X = dataset.iloc[:,0:-1].values
Y = dataset.iloc[:,-1].values
'''
X, Y = dataset.iloc[:, 1:-2].values, dataset.iloc[:, -1].values
# 划分测试集和训练集
seed = 7 # 随机因子，能保证多次的随机数据一致
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# 训练模型
model = XGBClassifier()
#model = XGBRegressor()
model.fit(X_train, y_train)
# 对模型做预测
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# 评估预测
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
'''
'''
df = pd.read_excel('F:/迁居文献/10-28语义识别结果/XGBOOST/类别0.xlsx')
x, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=666)#1
#rfc= XGBRegressor( random_state=666)# 15，500
rfc= XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=160, silent=True, objective='multi:softmax')# 15，500
rfc.fit(x_train, y_train)
y_pred=rfc.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print('accuracy:%2.f%%'%(accuracy*100))
 
# 显示重要特征
plot_importance(model)
plt.show()
#result=rfc.score(x_test,y_test)#score计算的是模型准确率
#print('准确率:%s' % result)
'''
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
 
# 导入数据集
df = pd.read_excel('F:/迁居文献/10-28语义识别结果/XGBOOST/类别0.xlsx')
x, y = df.iloc[:, 1:-1].values, df.iloc[:,-1].values
ss=preprocessing.MinMaxScaler()
x=ss.fit_transform(x)
y=ss.fit_transform(y)
# Xgboost训练过程
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

model = xgb.XGBRegressor(max_depth=5,learning_rate=0.1,n_estimators=160,silent=True,objective='reg:gamma')
model.fit(X_train,y_train)
 
# 对测试集进行预测
'''
y_pred= model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print('accuracy:%2.f%%'%(accuracy*100))
'''
# 显示重要特征
plot_importance(model)
plt.show()
