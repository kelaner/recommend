import math
import numpy as np
import random


# 制作测试集
def gene_demo_date(person, rate):
    # 生成学生评分
    stu_rate = np.zeros([person, rate], dtype=int)
    for num in range(person):
        for i in range(random.randint(2, 9)):
            stu_rate[num][random.randint(0, rate - 1)] = random.randint(1, 5)

    print(stu_rate[0])

    rates = []

    # 生成竞赛评分项目编号
    for i in range(65, 65 + rate):
        rates.append(chr(i))
    return stu_rate, rates


def similar(stu_rate, stu_id):
    # 欧氏距离计算,计算两者相似度
    def euclidean(a, b):
        a_date = stu_rate[a]
        b_date = stu_rate[b]
        distance = 0
        for x in range(len(a_date)):
            for y in range(len(b_date)):
                distance += pow(float(a_date[x]) - float(b_date[y]), 2)
        el = math.sqrt(distance)  # 值越大，相似度越大
        # print(el)
        # 返回值越小，相似度越大
        return 1 / (1 + el)

    res = []
    for p in range(len(stu_rate)):
        if stu_id == p:
            res.append(1.0)
        else:
            similar = euclidean(stu_id, p)
            res.append(similar)

    # print(res)

    # 获取相似度最高的用户
    sim_min_stu = res.index(min(res))
    # print(sim_min_stu)

    # 获取相似度最高的用户的评分记录
    items = list(stu_rate[sim_min_stu])

    recommends = []

    # 筛选出该用户未评分的竞赛并添加到列表中
    for i in range(len(items)):
        if stu_rate[stu_id][i] == 0 and items[i] != 0:
            recommends.append(chr(65 + i))

    print('第', stu_id+1, '位的推荐项目为:', recommends)


if __name__ == '__main__':
    stu_rate, rates = gene_demo_date(1000, 10)

    for num in range(5):
        similar(stu_rate, random.randint(0, 1000))
