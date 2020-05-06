import unittest

import matplotlib.pyplot as plt
import numpy as np

NOMIAL_2 = [0, 0, 2, -12, 9]
NOMIAL_4 = [0.005, 0, -2, 16, -9]
start_point = 30.0
alpha = 0.035 # 低步长保守
alpha = 0.085 # 高步长激进
MIN_TOLERANCE = 1e-6
plt.style.use('seaborn-dark-palette')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 梯度下降法
class GradTester(unittest.TestCase):
    # 绘制抛物线
    def test_parabola(self):
        draw_parabola(r"$2x^2-12x-9$", black_box_2)
        plt.legend()
        plt.show()

    # 绘制箭头
    def test_arrow2(self):
        draw_parabola(r"$2x^2-12x-9$", black_box_2)
        draw_arrow(10, 25, black_box_2)
        plt.legend()
        plt.show()

    # 绘制箭头
    def test_arrow4(self):
        draw_parabola(r"$0.005x^4-2x^2+16x-9$", black_box_4)
        draw_arrow(30, 10, black_box_4)
        plt.legend()
        plt.show()

    # 测试2次方的寻优，逼近唯一解
    def test_stochastic2(self):
        draw_parabola(r"$2x^2-12x-9$", black_box_2)  # 绘制完整的抛物线，从-20到+30
        cur_x = start_point
        for i in range(20000):
            # print("CurPoint %f" % cur_point)  # 打印当前点的x坐标
            plot_cur_point(cur_x, black_box_2)  # 绘制当前点，以*号标记
            diff = alpha * grad_black_box_2(cur_x)  # 计算x的移动距离=步长alpha*梯度ᐁ
            if abs(diff) <= MIN_TOLERANCE:  # 如果移动距离太小，说明逼近的足够近了，可以跳出循环
                print("AFTER %d ITERATIONS, PRECISION REACHED" % i)
                break
            next_x = cur_x - diff  # 按照梯度反方向步进
            draw_arrow(cur_x, next_x, black_box_2)  # 新旧point连线
            cur_x = next_x  # 从新的点往下继续
        plt.grid(True)
        plt.legend()
        plt.show()

    # 测试4次方的寻优，带局部最优
    def test_stochastic4(self):
        draw_parabola(r"$0.005x^4-2x^2+16x-9$", black_box_4)  # 绘制完整的抛物线，从-20到+30
        cur_x = start_point
        for i in range(20000):
            # print("CurPoint %f" % cur_point)  # 打印当前点的x坐标
            plot_cur_point(cur_x, black_box_4)  # 绘制当前点，以*号标记
            diff = alpha * grad_black_box_4(cur_x)  # 计算x的移动距离=步长alpha*梯度ᐁ
            if abs(diff) <= MIN_TOLERANCE:  # 如果移动距离太小，说明逼近的足够近了，可以跳出循环
                print("AFTER %d ITERATIONS, PRECISION REACHED" % i)
                break
            next_x = cur_x - diff  # 按照梯度反方向步进
            draw_arrow(cur_x, next_x, black_box_4)  # 新旧point连线
            cur_x = next_x  # 从新的点往下继续
        plt.grid(True)
        plt.legend()
        plt.show()


# 最高维为4的多项式
def orig_expo(x):
    return [x ** 4, x ** 3, x ** 2, x, 1]


# 最高维为4的多项式对应的导数
def gradient_expo(x):
    return [4 * x ** 3, 3 * x ** 2, 2 * x, 1, 0]


def multinomial_rep(coef, x):
    return sum([a * b for a, b in zip(coef, orig_expo(x))])


def grad_multinomial(coef, x):
    return sum([a * b for a, b in zip(coef, gradient_expo(x))])


def black_box_2(x):
    return multinomial_rep(NOMIAL_2, x)


def black_box_4(x):
    return multinomial_rep(NOMIAL_4, x)


def grad_black_box_2(x):
    return grad_multinomial(NOMIAL_2, x)


def grad_black_box_4(x):
    return grad_multinomial(NOMIAL_4, x)


def draw_parabola(label, algo):
    x = np.arange(-20, 30, 0.02)
    plt.plot(x, algo(x), c='black', lw=1, ls="--", label=label)


def draw_arrow(cur_point, next_point, algo):
    plt.annotate("", xy=(next_point, algo(next_point)), xytext=(cur_point, algo(cur_point)),
                 arrowprops=dict(arrowstyle="->", color="r", connectionstyle="arc3"))


def plot_cur_point(cur_point, algo):
    plt.scatter(cur_point, algo(cur_point), marker='*', c='navy', lw=1)
