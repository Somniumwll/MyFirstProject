import pandas as pd
import numpy as np


def f(): # 构建前80个x向量与y
    d1 = []
    d2= []
    i = 0
    while i < 80:
        x = []
        # 获取x1,x2,x3
        x1 = datas.iloc[i]["x1"]
        x2 = datas.iloc[i]["x2"]
        x3 = datas.iloc[i]["x3"]
        # 获取y
        y = datas.iloc[i]["y"]
        d2.append(y)
        # 将每行的x1,x2,x3作为一个x向量
        x.append(x1)
        x.append(x2)
        x.append(x3)
        d1.append(x)
        i += 1
    return d1, d2


def classify(datas, w, T):
    count = 0
    # 计算数据总数
    all = len(datas)
    datas = np.array(datas)
    for data in datas:
        # 判断是否分错
        # if np.dot(np.inner(data[0: 3], w.T), data[-1]) > 0:
        if np.dot(np.inner(w.T, data[-1]), data[0: 3]) > 0:
            # 若分对则count+1，计算对的数目
            count += 1
    print("当T为", T, "时，正确率为：", count/all)
    print("")


def my_perceptron(x, y, w):
    k = 0
    if np.dot(np.inner(w.T, x), y) <= 0:
        w = w + np.dot(y, x)
        k += 1
    return w


def my_execute(T):
    t = 0
    w = np.array([0, 0, 0])
    while t != T:
        for x, y in zip(xs, ys):
            w = my_perceptron(x, y, w)
            t += 1
            if t == T:
                break
        if t == T:
            break
    print("当T为", T, "时，w的值为", w)
    classify(datas[20:100], w, T)

if __name__ == '__main__':
    datas = pd.read_csv('/Users/apple/Desktop/linlin/data_x1x2x3y.csv')
    xs, ys = f()
    xsys = f()
    my_execute(200)  # T为200
    my_execute(50)  # T为50
    my_execute(20)  # T为20
    my_execute(5)  # T为5
