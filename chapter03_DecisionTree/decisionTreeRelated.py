import copy
import random
import unittest

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

MIN_ERR = 1e-8

ONE_PAIR = [i for i in range(1, 53)]
DIGIT_MAPPER = {1: "A", 11: "J", 12: "Q", 13: "K"}
for i in range(2, 11):
    DIGIT_MAPPER[i] = str(i)
SHAPE_MAPPER = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
FRONT_COLOR_MAPPER = {0: 36, 1: 35, 2: 31, 3: 33}
mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams['axes.unicode_minus'] = False


class PokerTester(unittest.TestCase):
    def test_pure_entropy(self):
        print(entropy(0.5))

    # 二元分类的熵，越分散熵越大
    def test_pure_entropylist(self):
        print(entropy_problist([0.5, 0.5]))
        print(entropy_problist([85, 85]))
        print(entropy_problist([850, 850]))
        print(entropy_problist([0.6, 0.4]))
        print(entropy_problist([0.7, 0.3]))
        print(entropy_problist([0.8, 0.2]))
        print(entropy_problist([0.9, 0.1]))

    # 绘制二元信息熵的图表
    def test_draw_binary_entropy(self):
        xx = []
        yy = []
        for x1 in range(1, 1000):
            one = x1 / 1000
            xx.append(one)
            lis = [one, 1.0 - one]
            yy.append(entropy_problist(lis))
        plt.plot(xx, yy, ls="-", lw=2, c="black")
        plt.grid(linestyle=":", color="b")
        plt.title("二元信息熵H(x)")
        parse_plot_grid_minor(plt)
        plt.show()

    # 多维参数
    def test_pure_entropylist_multi(self):
        print(entropy_problist([1]))
        print(entropy_problist([1, 1]))
        print(entropy_problist([1, 1, 1]))
        print(entropy_problist([1, 1, 1, 1]))
        print(entropy_problist([1, 1, 1, 1, 1]))
        print(entropy_problist([1 for v in range(128)]))

    def test_cards_no_cheat(self):
        cards = poker_gen(4, False)  # 2副牌
        arrs = split_to_n_pieces(cards, 4)  # 4个玩家
        for player_cards in arrs:
            player_cards = sorted(player_cards, key=sortKey)  # 按花色排序，如果按大小排序则去掉key
            print("-" * 100)
            for card in player_cards:
                print(decorate_color(pack_one_card(card), 1, FRONT_COLOR_MAPPER[judge_color(card)], 40),
                      end="")
            print()
            print("ENTROPY FOR THIS PLAYER IS %f Bit" % calc_arr_entropy(player_cards))

    def test_cards_with_cheat(self):
        cards = poker_gen(2, True)
        arrs = split_to_n_pieces(cards, 4)  # 4付牌
        for player_cards in arrs:
            player_cards = sorted(player_cards, key=sortKey)  # 按花色排序，如果按大小排序则去掉key
            print("-" * 100)
            for card in player_cards:
                print(decorate_color(pack_one_card(card), 1, FRONT_COLOR_MAPPER[judge_color(card)], 40),
                      end="")
            print("\nENTROPY FOR THIS PLAYER IS %f Bit" % calc_arr_entropy(player_cards))


def sortKey(digit):
    return ((judge_color(digit) + 1) << 6) + digit  # 颜色优先，然后digit


# 计算一副牌
def calc_arr_entropy(player_cards):
    color_dict = {}
    for card in player_cards:
        color = judge_color(card)
        color_dict[color] = 1 if color not in color_dict else color_dict[color] + 1
    reg = list(color_dict.values())
    reg = [r / len(player_cards) for r in reg]  # 归一化
    return entropy_problist(reg)


def decorate_color(raw, mode=0, front=31, back=40):
    complex_str = ('\033[%d;%d;%dm' + raw + '\033[0m ') % (mode, front, back)
    return complex_str


# cheat!!!
def poker_gen(n_pairs, cheat=False):
    count = 0
    while True:
        count += 1
        res = []
        for i in range(n_pairs):
            lis = copy.deepcopy(ONE_PAIR)
            random.shuffle(lis)  # 每副牌洗一遍
            res.extend(lis)
        random.shuffle(res)  # N副牌再洗一遍
        if cheat:
            first_p = split_to_n_pieces(res, 4)[0]
            ent = calc_arr_entropy(first_p)
            # print("CUR ENT=", ent)
            if ent < 1.5:  # 熵越小，牌越纯，当然也更难找
                break
        else:
            break
    if cheat:
        print("POKER GEN TRIED %d TIMES!" % count)
    return res


def split_to_n_pieces(full, n=4):
    arr = []
    cursor = 0
    step = int(len(full) / n)
    for i in range(n):
        arr.append(full[cursor: cursor + step])
        cursor += step
    return arr


def pack_one_card(digit):
    color = judge_color(digit)
    dig = int((digit - 1) >> 2) + 1
    shape = SHAPE_MAPPER[color] + DIGIT_MAPPER[dig]
    return shape


def judge_color(digit):
    return int((digit - 1) & 3)


# input [0.12, 0.21, ... 0.32]
# return shannon entropy
def entropy_problist(count_array):
    # assert sum(count_array) - 1.0 < MIN_ERR
    # assert len([element for element in count_array if element < 0]) == 0
    cc = [c / sum(count_array) for c in count_array]
    return sum([entropy(p) for p in cc])


def entropy(x):
    return -x * np.log2(x) if x > MIN_ERR else 0.0


# 进一步美化grid的展示
def parse_plot_grid_minor(plt, xgrid=0.05, ygrid=0.05):
    miloc_x = plt.MultipleLocator(xgrid)
    plt.gca().xaxis.set_minor_locator(miloc_x)
    plt.gca().grid(axis='x', which='minor', c="gray", ls="-.")
    miloc_y = plt.MultipleLocator(ygrid)
    plt.gca().yaxis.set_minor_locator(miloc_y)
    plt.gca().grid(axis='y', which='minor', c="gray", ls="-.")
    plt.grid(ls="--", lw=1.2, color="#666666")
