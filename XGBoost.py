import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score

def XGBoost(X, y, X_test, y_test):
    # 将数据转换为DMatrix格式
    dtrain = xgb.DMatrix(X, label=y)

    # 设置参数
    param = {
        'max_depth': 6,  # 树的最大深度
        'eta': 0.1,  # 学习率
        'objective': 'binary:logistic'  # 多分类问题的目标函数
        # 'num_class': 2  # 类别数量,当使用multi:softmax时取消注释
    }

    # 训练模型
    num_round = 1500  # 迭代次数
    model = xgb.train(param, dtrain, num_round)

    # 在训练集上进行预测
    train_pr = model.predict(dtrain)
    train_pred = (train_pr >= 0.5).astype(int)
    train_accuracy = accuracy_score(y, train_pred)
    print("XGB-训练集准确率：", train_accuracy)

    # 在测试集上进行预测
    dtest = xgb.DMatrix(X_test, label=y_test)
    y_pr = model.predict(dtest)
    y_pred = (y_pr >= 0.5).astype(int)

    # 计算模型的准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("XGB-测试集准确率:", accuracy)

    XGB_Plotting(y_test, X_test, y_pred, accuracy)

    return model

def XGB_Plotting(y_test, X_test, y_pred, accuracy):
    # 创建一个新的图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制测试集数据点
    scatter = ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], X_test.iloc[:, 2], c=y_test, cmap='viridis', label='True Labels')

    # 找到与y_test不一致的预测点的索引
    incorrect_indices = y_test != y_pred

    # 标记预测不一致的点为红色
    ax.scatter(X_test.loc[incorrect_indices, X_test.columns[0]], X_test.loc[incorrect_indices, X_test.columns[1]],
                c='red', marker='x', label='Incorrect Predictions')
    # 添加图例
    legend1 = ax.legend()
    ax.add_artist(legend1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f"XGB_Test_Accuracy: {accuracy}")
    plt.show()

if __name__ == '__main__':
    # 从 Excel 文件中读取数据集
    train_file_path = "./Dataset/TrainingData.xlsx"
    test_file_path = "./Dataset/TestData.xlsx"
    train_data = pd.read_excel(train_file_path)
    test_data = pd.read_excel(test_file_path)

    # 提取特征和标签
    X = train_data.iloc[:, :3]  # 前三列是特征（xyz值）
    y = train_data.iloc[:, 3]   # 第四列是标签（分类标签）

    #提取测试集
    X_test = test_data.iloc[:, :3]  # 前三列是特征（xyz值）
    y_test = test_data.iloc[:, 3]   # 第四列是标签（分类标签）

    XGB_model = XGBoost(X, y, X_test, y_test)