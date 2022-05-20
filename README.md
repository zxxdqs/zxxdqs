import pandas as pd
from sklearn import metrics
# 加载莺尾花数据集
from sklearn import datasets
# 导入高斯朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data = datasets.load_iris()
iris_target = data.target #得到数据对应的标签
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names) #利用Pandas转化为DataFrame格式
X_train, X_test, y_train, y_test = train_test_split(iris_features, iris_target, test_size=0.2, random_state=0)
# 使用高斯朴素贝叶斯进行计算
clf = GaussianNB()
clf.fit(X_train, y_train)
# 评估
test_predict = clf.predict(X_test)
print('The accuracy of the NB for Test Set is: %d%%' % (metrics.accuracy_score(y_test,test_predict)*100))
print(test_predict)
print(y_test)

# 预测
y_proba = clf.predict_proba(X_test[:1])
print(X_test[:1])
print(clf.predict(X_test[:1]))
print("预计的概率值:", y_proba)
