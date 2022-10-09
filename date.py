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

    # with tf.io.TFRecordWriter("./1.tfrecords") as writer:
    #     for i in range(10):
    #         example = tf.train.Example(features=tf.train.Features(feature={
    #             # "student": tf.train.Feature(float_list=tf.train.FloatList(value=[float]))
    #             "student": tf.train.Feature(int64_list=tf.train.Int64List(value=matrix3[i]))
    #         }))
    #         writer.write(example.SerializeToString())
    #
    # file = []
    # dataset = tf.data.TFRecordDataset("./1.tfrecords")
    # for i in range(10):
    #     for raw_record in dataset.take(i):
    #         example = tf.train.Example()
    #         example.ParseFromString(raw_record.numpy())
    #         print(example)

    # features = {
    #     'student': tf.FixedLenFeature((), tf.string),
    # }
    # parsed_features = tf.parse_single_example(dataset, features=features)
    # print(parsed_features)

    # for i in range(10):
    #     print(dataset[i], "\n")

    # def map_func(example):
    #     # feature 的属性解析表
    #     feature_map = {
    #                    'student': tf.FixedLenFeature((), tf.int64),
    #                    }
    #     parsed_example = tf.parse_single_example(example, features=feature_map)
    #
    #     # parsed_example["image"] 是 bytes 二进制数据，需要转化为 Tensor, 并告知转化后的 dtype
    #     # 我这里 np.random.rand 生成数据默认是 float64， 所以才这样写
    #     # 如果我们读取的图片数据是 0-255,则应该设置 out_type=tf.uint8
    #     image = tf.decode_raw(parsed_example["image"], out_type=tf.float64)
    #     height = parsed_example["height"]
    #     width = parsed_example["width"]
    #     channels = parsed_example["channels"]
    #
    #     # 我们将数据转化为 bytes, 再转化为张量, 会转化为一个 1维数据
    #     # 这里提前保存 shape 信息，转化回来
    #     image = tf.reshape(image, [height, width, channels])
    #     label = parsed_example["label"]
    #     return image, label
    #
    #
    # dataset = tf.data.TFRecordDataset(["./1.tfrecords"])
    # dataset = dataset.map(map_func=map_func)
    # iterator = dataset.make_one_shot_iterator()
    # element = iterator.get_next()
    #
    # with tf.Session() as sess:
    #     while True:
    #         try:
    #             image_, label_ = sess.run(element)
    #             print(image_.shape, image_.dtype, label_)  # (28, 28, 3) float64 1
    #         except OutOfRangeError:
    #             print("数据读取完毕")
    #             break
