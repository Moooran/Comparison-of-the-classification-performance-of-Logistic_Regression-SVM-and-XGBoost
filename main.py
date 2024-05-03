'''
比较利用Logistic Regression, SVM与XGBoost的分类性能。
其中SVM至少选用三种不同的Kernel Fucntions.
'''
import pandas as pd
from XGBoost import XGBoost
from Logistic_Regression import LR_model
from SVM import SVM_model



# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 从 Excel 文件中读取数据集
    train_file_path = "./Dataset/TrainingData.xlsx"
    test_file_path = "./Dataset/TestData.xlsx"
    train_data = pd.read_excel(train_file_path)
    test_data = pd.read_excel(test_file_path)

    # 提取特征和标签
    X = train_data.iloc[:, :3]  # 前三列是特征（xyz值）
    y = train_data.iloc[:, 3]  # 第四列是标签（分类标签）

    # 提取测试集
    X_test = test_data.iloc[:, :3]  # 前三列是特征（xyz值）
    y_test = test_data.iloc[:, 3]  # 第四列是标签（分类标签）

    print('{0:=<{1}}'.format('Logistic_Regression', 40))
    LR_model = LR_model(X, y, X_test, y_test)
    print('{0:=<{1}}'.format('SVM', 40))
    SVM_model = SVM_model(X, y, X_test, y_test)
    print('{0:=<{1}}'.format('XGBoost', 40))
    XGB_model = XGBoost(X, y, X_test, y_test)

