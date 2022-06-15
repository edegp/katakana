# coding: utf-8
import numpy as np


class SGD:
    """
    Stochastic Gradient Descent
    """
    def __init__(self, lr=0.01):
        """
        lr : 学習係数 learning rate
        """
        self.lr = lr
        
    def update(self, params, grads):
        """
        重みの更新
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            
class NesterovAG:
    """
    Nesterov Accelerated Gradient
    """
    def __init__(self, lr=0.01, momentum=0.9):
        """
        lr : 学習係数 learning rate
        momentm : モーメンタム係数
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        """
        重みの更新
        """
        if self.v is None:
            """
            初回のみ
            """
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        # 重みを更新
        for key in params.keys():
            v_pre = self.v[key].copy()
            self.v[key] = v_pre * self.momentum - self.lr * grads[key]
            params[key] += -self.momentum* v_pre + (self.momentum+1) * self.v[key]


class RMSProp:
    """
    RMSProp
    """
    def __init__(self, lr=0.01, rho=0.9):
        """
        lr : 学習係数 learning rate
        rho : 減衰率
        """
        self.lr = lr
        self.h = None
        self.rho = rho
        self.epsilon = 1e-6
        
    def update(self, params, grads):
        """
        重みの更新
        """
        if self.h is None:
            """
            初回のみ
            """
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.h[key] = self.rho * self.h[key] + (1 - self.rho) * grads[key] * grads[key]          
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key] + self.epsilon) ) # 原論文に合わせてepsilonをルートの中に入れる


class RMSPropGraves:
    """
    RMSProp
    """
    def __init__(self, lr=0.0001, rho=0.95):
        """
        lr : 学習係数 learning rate
        rho : 減衰率
        """
        self.lr = lr
        self.m = None  # 1次モーメント. 勾配の平均に相当する
        self.v = None  # 2次モーメント. 勾配の分散に相当する(中心化されていない)
        self.rho = rho
        self.epsilon = 1e-4
        
    def update(self, params, grads):
        """
        重みの更新
        """
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

                
        for key in params.keys():
            self.m[key]  = self.rho * self.m[key]  + (1 - self.rho) * grads[key]
            self.v[key] = self.rho * self.v[key] + (1 - self.rho) * grads[key] * grads[key]
            # 原論文に合わせてepsilonをルートの中に入れる
            params[key] -= self.lr * grads[key] / (np.sqrt(self.v[key] - self.m[key] * self.m[key] + self.epsilon))


class Adam:
    """
    Adam
    """

    def __init__(self, lr=0.001, rho1=0.9, rho2=0.999):
        self.lr = lr
        self.rho1 = rho1
        self.rho2 = rho2
        self.iter = 0
        self.m = None  # 1次モーメント. 勾配の平均に相当する
        self.v = None  # 2次モーメント. 勾配の分散に相当する(中心化されていない)
        self.epsilon = 1e-8

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1

        for key in params.keys():
            # print(grads)
            self.m[key] = self.rho1*self.m[key] + \
                (1-self.rho1)*grads[key]  # 1次モーメント
            self.v[key] = self.rho2*self.v[key] + \
                (1-self.rho2)*(grads[key]**2)  # 2次モーメント

            # モーメントのバイアス補正
            # 計算初期の頃のモーメントを補正することが目的
            # 計算が進行する(self.iterが大きくなる)と、分母は1に近づく
            m = self.m[key] / (1 - self.rho1**self.iter)  # 1次モーメント
            v = self.v[key] / (1 - self.rho2**self.iter)  # 2次モーメント

            # 重みの更新
            params[key] -= self.lr * m / (np.sqrt(v) + self.epsilon)


class AdamW:
    """
    Adam
    """

    def __init__(self, lr=0.001, rho1=0.9, rho2=0.999, wd=1e-3):
        self.lr = lr
        self.rho1 = rho1
        self.rho2 = rho2
        self.iter = 0
        self.m = None  # 1次モーメント. 勾配の平均に相当する
        self.v = None  # 2次モーメント. 勾配の分散に相当する(中心化されていない)
        self.epsilon = 1e-8
        self.wd= wd

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1

        for key in params.keys():
            params[key] -= self.lr * self.wd * params[key]
            self.m[key] = self.rho1*self.m[key] + \
                (1-self.rho1)*grads[key]  # 1次モーメント
            self.v[key] = self.rho2*self.v[key] + \
                (1-self.rho2)*(grads[key]**2)  # 2次モーメント

            # モーメントのバイアス補正
            # 計算初期の頃のモーメントを補正することが目的
            # 計算が進行する(self.iterが大きくなる)と、分母は1に近づく
            m = self.m[key] / (1 - self.rho1**self.iter)  # 1次モーメント
            v = self.v[key] / (1 - self.rho2**self.iter)  # 2次モーメント

            # 重みの更新
            params[key] -= self.lr * m / (np.sqrt(v) + self.epsilon)

class Adadelta:
    """
    Adadelta
    """

    def __init__(self, rho=0.75):
        """
        rho : 減衰率
        """
        self.h = None
        self.r = None
        self.rho = rho
        self.epsilon = 1e-6

    def update(self, params, grads):
        """
        重みの更新
        """
        if self.h is None:
            """
            初回のみ
            """
            self.h = {}
            self.r = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                self.r[key] = np.zeros_like(val)

        for key in params.keys():

            # 1ステップ前における更新量の2乗の移動平均のルートを求める
            # <- 穴埋め
            rms_param = (self.r[key] + self.epsilon)**0.5

            # 勾配の2乗の移動平均を求める
            # <- 穴埋め
            self.h[key] = self.rho*self.h[key]+(1-self.rho)*grads[key]**2

            # 勾配の2乗の移動平均のルートを求める
            # <- 穴埋め
            rms_grad = (self.h[key] + self.epsilon)**0.5

            # 更新量の算出
            # <- 穴埋め
            dp = -rms_param/rms_grad*grads[key]

            # 重みの更新
            params[key] += dp

            # 次ステップのために、更新量の2乗の移動平均を求める
            # <- 穴埋め
            self.r[key] = self.rho*self.r[key] + (1-self.rho)*(dp)**2

