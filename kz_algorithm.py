import math


# Jac相似算法
def jac(obj_student, obj_standard):
    same = 0
    if len(obj_student) != len(obj_standard):
        return -1
    for row in range(len(obj_standard)):
        if obj_standard[row] == obj_student[row]:
            same += 1
    score = same / len(obj_standard)
    return score - 0.5


# Sim余弦算法
def sim(a, b, fix):
    if len(a) != len(b):
        return -1
    average_a = param_fix(a, fix)
    # print(average_a[30])
    average_b = param_fix(b, fix)
    nume = 0  # 分子
    for row in range(len(average_a)):
        nume += average_a[row] * average_b[row]
    deno = absolute_all(average_a) * absolute_all(average_a)
    score = 0.5 + 0.5 * (nume / (deno + float("1e-8")))
    return score


# 修正系数参数求和
def param_fix(arr, fix):
    for num in range(len(arr)):
        arr[num] = arr[num] * fix[num]
    return arr


# 绝对值求和
def absolute_all(arr):
    total = 0
    for num in range(len(arr)):
        total += arr[num] ** 2
    output = math.sqrt(total)
    return output


# Cov协方差算法
def cov(a, b, fix):
    if len(a) != len(b):
        return -1
    average_a = param_fix(a, fix)
    average_b = param_fix(b, fix)
    sum_ab = 0
    sqrt_a = 0
    sqrt_b = 0
    for row in range(len(a)):
        sum_ab += (average_a[row] - average_all(average_a)) * (average_b[row] - average_all(average_b))
        sqrt_a += (average_a[row] - average_all(average_a)) ** 2
        sqrt_b += (average_b[row] - average_all(average_b)) ** 2
    sqrt_a = math.sqrt(sqrt_a)
    sqrt_b = math.sqrt(sqrt_b)
    score = sum_ab / (sqrt_a * sqrt_b + float("1e-8"))
    return score


# 平均数求和
def average_all(arr):
    total = 0
    for num in range(len(arr)):
        total += arr[num]
    output = total / len(arr)
    return output

# # 时间缓释函数
# def time_out(now_time, stand_time):
#     alpha = 1
#     return math.pow(math.e, alpha * (now_time - stand_time))
