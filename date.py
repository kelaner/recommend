# import tensorflow as tf
import random
import numpy as np
import kz_algorithm
import matplotlib.pyplot as plt

plt.ion()


# 通过随机函数构建学生与竞赛数据集，使用tfrecord保存数据集
def base_date(person, rate, n):
    stu = []  # 构建学生列表
    sta = []  # 构建竞赛的理想学生

    # 生成学生特征因子
    def generate_student():
        a = []
        for num in range(n):
            # 随机 0~1 因子
            a.append(round(float(random.random()), 1))

            # 二分法因子
            # i = round(float(random.random()), 1)
            # if i > 0.5:
            #     a.append(1)
            # else:
            #     a.append(0)

        stu.append(a)

    for num in range(person):
        generate_student()

    # # 输出学生列表
    # for num in range(len(stu)):
    #     print(stu[num], '\n')

    # 生成理想学生特征因子
    def generate_standard():
        a = []
        for num in range(n):
            # 随机 0~1 因子
            a.append(round(float(random.random()), 1))

            # 二分法因子
            # i = round(float(random.random()), 1)
            # if i > 0.5:
            #     a.append(1)
            # else:
            #     a.append(0)

        sta.append(a)

    for num in range(rate):
        generate_standard()

    # # 输出理想学生列表
    # for num in range(len(sta)):
    #     print(sta[num], '\n')

    return stu, sta


# 生成学生对竞赛的二维矩阵
def two_dim_date(stu, sta, s):
    # 构建学生与竞赛推荐矩阵
    # stu_sta = [[0 for _ in range(len(sta))] for _ in range(len(stu))]
    stu_sta = np.zeros([len(stu), len(sta)], dtype=int)

    # 构建学生与竞赛多因子匹配矩阵
    # matrix = [[0 for _ in range(len(sta))] for _ in range(len(stu))]
    matrix = np.zeros([len(stu), len(sta)], dtype=float)

    # 修正因子设定
    fix = []
    for param in range(len(stu)):
        fix.append(1.)

    for i in list(np.random.randint(15000, size=300)):
        fix[i] = round(float(random.random()), 1)

    # 算法调用
    for rate in range(len(sta)):
        for person in range(len(stu)):
            if s == 'cov':
                score = kz_algorithm.cov(stu[person], sta[rate], fix)
            elif s == 'sin':
                # 加入二分法更符合预期效果
                for i in range(len(stu[person])):
                    if stu[person][i] > 0.5:
                        stu[person][i] = 1
                    else:
                        stu[person][i] = 0
                for i in range(len(sta[rate])):
                    if sta[rate][i] > 0.5:
                        sta[rate][i] = 1
                    else:
                        sta[rate][i] = 0
                score = kz_algorithm.sim(stu[person], sta[rate], fix)
            elif s == 'jac':
                # 加入二分法更符合预期效果
                for i in range(len(stu[person])):
                    if stu[person][i] > 0.5:
                        stu[person][i] = 1
                    else:
                        stu[person][i] = 0
                for i in range(len(sta[rate])):
                    if sta[rate][i] > 0.5:
                        sta[rate][i] = 1
                    else:
                        sta[rate][i] = 0
                score = kz_algorithm.jac(stu[person], sta[rate])
            else:
                score = 0
                print("no choice")
            if score > 0:
                stu_sta[person][rate] = 1
            matrix[person][rate] = score

    # # 输出多因子匹配矩阵
    # for num in range(len(matrix)):
    #     print(matrix[num], '\n')

    return stu_sta, matrix


# 可视化
def show(m):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    x = range(15000)
    y = list(map(list, zip(*m)))[1]  # 矩阵行列变换后选定具体竞赛列表化
    plt.figure()
    # plt.ylim(0.0, 1.0)  # 限制y轴范围
    plt.plot(x, y, "-b", linewidth=0.2)


if __name__ == '__main__':
    n = 15  # 特征数
    stu, sta = base_date(15000, 10, n)
    stu_sta1, matrix1 = two_dim_date(stu, sta, 'cov')
    show(matrix1)
    stu_sta2, matrix2 = two_dim_date(stu, sta, 'sin')
    show(matrix2)
    stu_sta3, matrix3 = two_dim_date(stu, sta, 'jac')
    show(matrix3)
    plt.ioff()
    plt.show()

    # 保存推荐矩阵
    np.savetxt("stu_sta.csv", stu_sta1, delimiter=",", fmt="%d")


