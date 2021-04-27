#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import tm
import time
import numpy as np
import pandas as pd

y = np.random.randint(2, size=(100000, 1))
x = np.random.randint(10, size=(100000, 1))
data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])

print("input_y_shape:{}\ninput_x_shape{}".format(y.shape, x.shape))


def v2():
    start = time.time()
    result_v2 = tm.target_mean_v2(data, 'y', 'x')
    print("原始运行时间：{}".format(time.time() - start))
    return result_v2


def v3():
    start = time.time()
    result_v3 = tm.target_mean_v3(data, 'y', 'x')
    print("教师版运行时间：{}".format(time.time() - start))
    return result_v3


def v4():
    start = time.time()
    result_v4 = tm.target_mean_v4(data, 'y', 'x')
    print("学生版运行时间：{}".format(time.time() - start))
    return result_v4


result_2 = v2()
result_3 = v3()
result_4 = v4()

diff = np.linalg.norm(result_2 - result_3)
print("原始结果与教师版数据比对:{}".format(diff))
diff = np.linalg.norm(result_2 - result_4)
print("原始结果与学生版数据比对:{}".format(diff))
diff = np.linalg.norm(result_3 - result_4)
print("教师版与学生版数据比对:{}".format(diff))
