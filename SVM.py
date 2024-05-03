'''
kernel参数可以设置为不同的核函数类型，常用的核函数包括：
线性核函数（Linear Kernel）：线性核函数通过计算输入特征和支持向量之间的线性关系来进行分类，即内积。在Scikit-learn中，你可以将kernel='linear'来指定线性核函数。
多项式核函数（Polynomial Kernel）：多项式核函数通过计算输入特征和支持向量之间的多项式关系来进行分类。你可以指定kernel='poly'来选择多项式核函数，并可以通过degree参数来控制多项式的阶数。
径向基核函数（Radial Basis Function Kernel，RBF Kernel）：径向基核函数是SVM中最常用的核函数之一，默认情况下也是Scikit-learn中的默认核函数。径向基核函数通过计算输入特征和支持向量之间的距离来进行分类。在Scikit-learn中，你可以将kernel='rbf'来指定径向基核函数。
sigmoid核函数（Sigmoid Kernel）：sigmoid核函数通过计算输入特征和支持向量之间的sigmoid函数关系来进行分类。你可以指定kernel='sigmoid'来选择sigmoid核函数。
除了这些常用的核函数，还有其他一些核函数可供选择，如高斯核函数（Gaussian Kernel）、拉普拉斯核函数（Laplacian Kernel）等。
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

def SVM_model(x, y, X_test, y_test):
    # 创建SVM模型对象
    svm_model1 = svm.SVC(kernel='linear')  # 使用线性内核
    svm_model2 = svm.SVC(kernel='poly')  # 使用多项式内核，阶数为3
    svm_model3 = svm.SVC(kernel='rbf')  # 使用高斯径向基函数内核，gamma为0.1
    svm_model4 = svm.SVC(kernel='sigmoid')  #使用sigmoid核函数

    #训练各个kernel模型
    train_model(svm_model1, 'linear', x, y, X_test, y_test)
    train_model(svm_model2, 'poly', x, y, X_test, y_test)
    train_model(svm_model3, 'rbf', x, y, X_test, y_test)
    train_model(svm_model4, 'sigmoid', x, y, X_test, y_test)

def train_model(model, func, x, y, X_test, y_test):
    # 训练模型
    model.fit(x, y)

    # 在训练集上进行预测
    train_pred = model.predict(x)
    train_accuracy = accuracy_score(y, train_pred)
    print("SVM-{}-训练集准确率：{}".format(func, train_accuracy))

    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("SVM-{}-测试集准确率：{}".format(func, test_accuracy))

    SVM_Plotting(y_test, X_test, y_pred, test_accuracy, func)

    return model

def SVM_Plotting(y_test, X_test, y_pred, accuracy, func):
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
    plt.title("SVM_{}_Accuracy:{}".format(func, accuracy))
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

    # 提取测试集
    X_test = test_data.iloc[:, :3]  # 前三列是特征（xyz值）
    y_test = test_data.iloc[:, 3]  # 第四列是标签（分类标签）

    SVM_model = SVM_model(X, y, X_test, y_test)



