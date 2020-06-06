#!/usr/bin/env python
#-*- coding = utf-8 -*-
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

#梯度提升回归（虽然叫回归，实际上既可以回归，也可以分类）
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1, learning_rate=0.01)
#定义树的深度max_depth，防止过拟合；  学习率用于控制每一棵树纠正前一棵树的错误的强度
gbrt.fit(X_train,y_train)

print("Accuracy on trainging set:{:.3f}".format(gbrt.score(X_train,y_train)))
print("Accuracy on test set:{:.3f}".format(gbrt.score(X_test,y_test)))

#查看树的重要特征排序
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1] #获取data的第2个数，即特征的个数
    plt.barh(range(n_features), model.feature_importances_, align = 'center')#绘制条形图
    plt.yticks(np.arange(n_features),cancer.feature_names) #显示x轴的刻标、及对应的标签
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

plot_feature_importances_cancer(gbrt)
plt.show()