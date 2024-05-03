import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def LR_model(X, y, X_test, y_test):
    # 定义并训练 Logistic Regression 模型
    model = LogisticRegression(max_iter=100, tol=0.1, solver='liblinear', penalty='l1', C=0.01)
    model.fit(X, y)

    # 在训练集上进行预测
    train_pred = model.predict(X)
    train_accuracy = accuracy_score(y, train_pred)
    print("Logistic_Regression训练集准确率：", train_accuracy)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Logistic_Regression测试集准确率：", test_accuracy)

    LR_Plotting(y_test, X_test, y_pred, test_accuracy)

    return model

def LR_Plotting(y_test, X_test, y_pred, accuracy):
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
    plt.title(f"LR_Test_Accuracy: {accuracy}")
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


    LR_model = LR_model(X, y, X_test, y_test)
