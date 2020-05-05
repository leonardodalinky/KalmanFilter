'''
Kalman-Filter

对于一个线性系统的状态差分方程为
    x_k = A * x_{k-1} + B*u_{k-1} + w_k
其中x是系统的状态向量，大小为n*1列。A为转换矩阵，大小为n*n。
u为系统输入，大小为k*1。B是将输入转换为状态的矩阵，大小为n*k。
随机变量w为系统噪声。注意这些矩阵的大小，它们与你实际编程密切相关。

测量值当然是由系统状态变量映射出来的，方程形式如下：
    z_k = H*x_k + v_k
注意，此处的x_k表示的是仍是上面的系统预测值，这个测量值的映射只不过是理想情况下。
注意Z是测量值，大小为m*1(m取决于测量值），H也是状态变量到测量的转换矩阵。大小为m*n。随机变量v是测量噪声。

首先要计算预测值、预测值和真实值之间误差协方差矩阵。
    (1) \hat{x_k^{-}} = A * \hat{x_{k-1}} + B * u_{k-1}
    (2) P_k^{-} = A * P_{k-1} * A^T + Q
其中 \hat{x_k^{-}} 和 P_k^{-} 表示预测的结果和预测的协方差矩阵。
Q 表示系统噪声

有了这两个就能计算卡尔曼增益K，再然后得到估计值，
    (3) K_k = P_k^{-} * H^T * (H * P_k^{-} * H^T + R)^(-1)
    (4) \hat{x_k} = \hat{x_k^{-}} + K_k * (z_k - H * \hat{x_k^{-}})
其中 K_k 为卡尔曼增益，R 表示传感器噪声

最后还要计算估计值和真实值之间的误差协方差矩阵，为下次递推做准备。
    (5) P_k = (I - K_k * H) * P_k^{-}
I 表示单位矩阵
'''
import numpy as np
from typing import Tuple


def KM(x: np.matrix, u: np.matrix, z: np.matrix, A: np.matrix, B: np.matrix, P: np.matrix,
       H: np.matrix, Q: np.matrix, R: np.matrix) -> Tuple[np.matrix, np.matrix]:
    '''
    执行 Kalman filter 的一个过程
    :param x: 上一次的实际结果
    :param u: 系统输入, 大小为k*1。可设置为None
    :param z: 此次的测量值, 大小为m*1
    :param A: 系统转换矩阵, 大小为n*n
    :param B: 将输入转换为状态的矩阵, 大小为n*k
    :param P: 上一次的估计值与真实值之间的协方差矩阵, 大小为n*n
    :param H: 理想状态下，状态变量到测量的转换矩阵, 大小为m*n
    :param Q: 系统噪声, 大小为n*n
    :param R: 传感器噪声, 大小为m*m
    :return: 返回的第一个量为此次的估计值，第二个量为此次的协方差矩阵
    '''
    # (1)
    hat_x_k_: np.matrix = None
    if B is not None and u is not None:
        hat_x_k_ = A @ x + B @ u
    else:
        hat_x_k_ = A @ x

    # (2)
    P_k_: np.matrix = None
    if Q is not None:
        P_k_ = A @ P @ A.T + Q
    else:
        P_k_ = A @ P @ A.T

    # (3)
    K_k: np.matrix = None
    if R is not None:
        K_k = P_k_ @ H.T @ np.linalg.inv(H @ P_k_ @ H.T + R)
    else:
        K_k = P_k_ @ H.T @ np.linalg.inv(H @ P_k_ @ H.T)

    # (4)
    hat_x_k: np.matrix = None
    hat_x_k = hat_x_k_ + K_k @ (z - H @ hat_x_k_)

    # (5)
    temp: np.matrix = K_k @ H
    I: np.matrix = np.mat(np.identity(temp.shape[0]))
    P_k: np.matrix = (I - temp) @ P_k_

    return hat_x_k, P_k
