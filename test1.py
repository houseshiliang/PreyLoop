import numpy as np
from sklearn.neighbors import KernelDensity

def h():
    return 3

if __name__=="__main__":

    # 创建一个示例的三维数组
    arr = np.random.random((2, 4, 3))  # 一个形状为 (4, 3, 5) 的示例数组

    # 使用 numpy.split() 对第三维进行分割
    result = np.dsplit(arr, 3)
    print(result)
    print(result[0])

