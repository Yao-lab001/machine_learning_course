import unittest
from ctypes import c_float, c_int32, cast, byref, POINTER

# 注意其返回的是平方根的倒数，省掉一个倒数计算速度会提升很快
# python转写c语言的方法参见https://ajcr.net/fast-inverse-square-root-python/
def ctypes_isqrt(number):
    x2 = number * 0.5
    y = c_float(number)
    i = cast(byref(y), POINTER(c_int32)).contents.value
    i = c_int32(0x5f3759df - (i >> 1))
    y = cast(byref(i), POINTER(c_float)).contents.value
    y = y * (1.5 - (x2 * y * y))
    return y

class NbTester(unittest.TestCase):
    def test_inv_1(self):
        print("sqrt of 121 is " , 1.0 / ctypes_isqrt(121.0))
        print("sqrt of 1210 is " , 1.0 / ctypes_isqrt(1210.0))
        print("sqrt of 1210 is " , 1.0 / ctypes_isqrt(12100.0))