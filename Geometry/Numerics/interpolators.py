from .mutils import clamp, compute_derivatives_2_at_pt, compute_derivatives_2
from Geometry.common import NUMERICAL_ACCURACY, fast_math, parallel_range, parallel
from typing import Tuple
from cmath import sqrt
import numpy as np


@fast_math
def bi_linear_interp_pt(x: float, y: float, points: np.ndarray, width: float = 1.0, height: float = 1.0) -> float:
    """
    Билинейная интерполяция точки (x,y).
    :param x: x - координата точки.
    :param y: y - координата точки.
    :param points: одномерный список узловых точек.
    :param width: ширина области интерполяции.
    :param height: высота области интерполяции.
    :return:
    """
    if points.ndim != 2:
        print("bi_linear_interp_pt :: points array has to be 2 dimensional")
        return 0.0
        # raise RuntimeError("bi_linear_interp_pt :: points array has to be 2 dimensional")

    rows, cols = points.shape[0], points.shape[1]

    x = clamp(x, 0.0, width)

    y = clamp(y, 0.0, height)

    col_ = int((x / width) * (cols - 1))

    row_ = int((y / height) * (rows - 1))

    col_1 = min(col_ + 1, cols - 1)

    row_1 = min(row_ + 1, rows - 1)

    # q11 = nodes[row_, col_]

    # q00____q01
    # |       |
    # |       |
    # q10____q11

    dx_ = width / (cols - 1.0)
    dy_ = height / (rows - 1.0)

    tx = (x - dx_ * col_) / dx_
    ty = (y - dy_ * row_) / dy_

    q00: float = points[row_,  col_]
    q01: float = points[row_,  col_1]
    q10: float = points[row_1, col_]
    q11: float = points[row_1, col_1]

    return q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11)


@fast_math
def bi_linear_interp_derivatives_pt(x: float, y: float, points: np.ndarray, width: float = 1.0,
                                    height: float = 1.0, dx: float = 0.001, dy: float = 0.001) -> \
        Tuple[float, float]:
    dx = width * dx
    dy = height * dy

    f_p_dx = bi_linear_interp_pt(x + dx, y, points, width, height)
    f_m_dx = bi_linear_interp_pt(x - dx, y, points, width, height)
    f_p_dy = bi_linear_interp_pt(x, y + dy, points, width, height)
    f_m_dy = bi_linear_interp_pt(x, y - dy, points, width, height)
    return (f_p_dx - f_m_dx) * 0.5 / dx, (f_p_dy - f_m_dy) * 0.5 / dy


@fast_math
def bi_linear_interp_derivatives2_pt(x: float, y: float, points: np.ndarray, width: float = 1.0,
                                     height: float = 1.0, dx: float = 0.001, dy: float = 0.001) -> \
        Tuple[float, float, float]:
    dx = width * dx
    dy = height * dy

    f_p_dx = bi_linear_interp_pt(x + dx, y, points, width, height)
    f_m_dx = bi_linear_interp_pt(x - dx, y, points, width, height)
    f_p_dy = bi_linear_interp_pt(x, y + dy, points, width, height)
    f_m_dy = bi_linear_interp_pt(x, y - dy, points, width, height)

    f_p_dy_dx = bi_linear_interp_pt(x + dx, y + dy, points, width, height)
    f_m_dy_dx = bi_linear_interp_pt(x - dx, y - dy, points, width, height)
    return (f_p_dx - f_m_dx) * 0.5 / dx, \
           (f_p_dy - f_m_dy) * 0.5 / dy, \
           (f_p_dy_dx - f_p_dx - f_p_dy + f_m_dy_dx) * 0.25 / dx / dy


@parallel
def bi_linear_interp_derivatives(x: np.ndarray, y: np.ndarray, points: np.ndarray, width: float = 1.0,
                                 height: float = 1.0, dx: float = 0.001, dy: float = 0.001) -> np.ndarray:
    """
    Би-линейная интерполяция частных производных диапазона точек x, y.
    :param x: x - координаты точек.
    :param y: y - координаты точек.
    :param points: одномерный список узловых точек.
    :param width: ширина области интерполяции.
    :param height: высота области интерполяции.
    :param dx:
    :param dy:
    :return:
    """
    result = np.zeros((y.size, x.size, 2), dtype=float)
    for i in parallel_range(result.size // 2):
        res_row_, res_col_ = divmod(i, x.size)
        _x, _y = bi_linear_interp_derivatives_pt(x[res_col_], y[res_row_], points, width, height, dx, dy)
        result[res_row_, res_col_, 0] = _x
        result[res_row_, res_col_, 1] = _y
    return result


@parallel
def bi_linear_interp_derivatives2(x: np.ndarray, y: np.ndarray, points: np.ndarray, width: float = 1.0,
                                  height: float = 1.0, dx: float = 0.001, dy: float = 0.001) -> np.ndarray:
    """
    Би-линейная интерполяция частных производных диапазона точек x, y.
    :param x: x - координаты точек.
    :param y: y - координаты точек.
    :param points: одномерный список узловых точек
    :param width: ширина области интерполяции.
    :param height: высота области интерполяции.
    :param dx:
    :param dy:
    :return:
    """
    result = np.zeros((y.size, x.size, 3), dtype=float)
    for i in parallel_range(result.size // 3):
        res_row_, res_col_ = divmod(i, x.size)
        _x, _y, _z = bi_linear_interp_derivatives2_pt(x[res_col_], y[res_row_], points, width, height, dx, dy)
        result[res_row_, res_col_, 0] = _x
        result[res_row_, res_col_, 1] = _y
        result[res_row_, res_col_, 2] = _z
    return result


@parallel
def bi_linear_interp(x: np.ndarray, y: np.ndarray, points: np.ndarray,
                     width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """
    Билинейная интерполяция диапазона точек x, y.
    :param x: x - координаты точек.
    :param y: y - координаты точек.
    :param points: одномерный список узловых точек.
    :param width: ширина области интерполяции.
    :param height: высота области интерполяции.
    :return:
    """
    if points.ndim != 2:
        print("bi_linear_interp_pt :: points array has to be 2 dimensional")
        return np.array([0.0])
        # raise RuntimeError("bi_linear_interp_pt :: points array has to be 2 dimensional")

    rows, cols = points.shape[0], points.shape[1]

    result = np.zeros((y.size, x.size,), dtype=float)

    dx_ = width / (cols - 1.0)

    dy_ = height / (rows - 1.0)

    for i in parallel_range(result.size):
        res_row_, res_col_ = divmod(i, x.size)

        x_ = clamp(x[res_col_], 0.0, width)

        y_ = clamp(y[res_row_], 0.0, height)

        col_ = int((x_ / width) * (cols - 1))

        row_ = int((y_ / height) * (rows - 1))

        col_1 = min(col_ + 1, cols - 1)

        row_1 = min(row_ + 1, rows - 1)

        # q11 = nodes[row_, col_]
        # q00____q01
        # |       |
        # |       |
        # q10____q11

        tx = (x_ - dx_ * col_) / dx_
        ty = (y_ - dy_ * row_) / dy_

        q00: float = points[row_,  col_]
        q01: float = points[row_,  col_1 ]
        q10: float = points[row_1, col_]
        q11: float = points[row_1, col_1]

        result[res_row_, res_col_] = q00 + (q01 - q00) * tx + (q10 - q00) * ty + tx * ty * (q00 - q01 - q10 + q11)

    return result


@parallel
def bi_linear_cut(x_0: float, y_0: float, x_1: float, y_1: float, steps_n: int, points: np.ndarray,
                  width: float = 1.0, height: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Сечение интерполируемой, би-линейным методом поверхности вдоль прямой через две точки.
    :param x_0: x - координата первой точки секущей.
    :param y_0: y - координата первой точки секущей.
    :param x_1: x - координата второй точки секущей.
    :param y_1: y - координата второй точки секущей.
    :param steps_n: количество точке вдоль секущей.
    :param points: одномерный список узловых точек.
    :param width: ширина области интерполяции.
    :param height: высота области интерполяции.
    :return:
    """
    dx = x_1 - x_0
    dy = y_1 - y_0
    rho = dx * dx + dy * dy

    if rho > NUMERICAL_ACCURACY:
        rho = sqrt(rho)
        dx /= rho
        dy /= rho
    else:
        dx = 0.0
        dy = 0.0

    dt = 1.0 / (steps_n - 1)

    points_x = np.zeros((steps_n,), dtype=float)

    points_y = np.zeros((steps_n,), dtype=float)

    points_fxy = np.zeros((steps_n,), dtype=float)

    for i in parallel_range(steps_n):
        points_x[i] = dt * dx * i + x_0
        points_y[i] = dt * dy * i + y_0
        points_fxy[i] = bi_linear_interp_pt(points_x[i], points_y[i], points, width, height)

    return points_x, points_y, points_fxy


@parallel
def bi_linear_cut_along_curve(x_pts: np.ndarray, y_pts: np.ndarray, points: np.ndarray,
                              width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """
    Сечение интерполируемой би-линейным методом поверхности вдоль кривой, заданной в виде массива точек.
    :param x_pts: координаты кривой по х.
    :param y_pts: координаты кривой по y.
    :param points: одномерный список узловых точек.
    :param width: ширина области интерполяции.
    :param height: высота области интерполяции.
    :return:
    """
    cut_values = np.zeros((min(x_pts.size, y_pts.size),), dtype=float)
    for i in parallel_range(cut_values.size):
        cut_values[i] = bi_linear_interp_pt(x_pts[i], y_pts[i], points, width, height)
    return cut_values


@fast_math
def _cubic_poly(x: float, y: float, m: np.ndarray) -> float:
    """
    Вспомогательная функция для вычисления кубического полинома би кубической интерполяции.
    :param x: x - координата.
    :param y: y - координата.
    :param m: матрица коэффициентов.
    :return:
    """
    x2 = x * x
    x3 = x2 * x
    y2 = y * y
    y3 = y2 * y
    return (m[0] + m[1] * y + m[2] * y2 + m[3] * y3) + \
           (m[4] + m[5] * y + m[6] * y2 + m[7] * y3) * x + \
           (m[8] + m[9] * y + m[10] * y2 + m[11] * y3) * x2 + \
           (m[12] + m[13] * y + m[14] * y2 + m[15] * y3) * x3


@fast_math
def _bi_qubic_interp_pt(x: float, y: float, points: np.ndarray, points_dx: np.ndarray,
                        points_dy: np.ndarray, points_dxy: np.ndarray,
                        width: float = 1.0, height: float = 1.0) -> float:
    """
    :param x: координата точки по х.
    :param y: координата точки по y.
    :param points: одномерный список узловых точек.
    :param points_dx: производная по х в узловых точках.
    :param points_dy: производная по y в узловых точках.
    :param points_dxy: производная по хy в узловых точках.
    :param width: ширина области интерполяции.
    :param height: высота области интерполяции.
    :return:
    """
    rows, cols = points.shape[0], points.shape[1]

    x = clamp(x, 0.0, width)

    y = clamp(y, 0.0, height)

    col_ = int((x / width) * (cols - 1))

    row_ = int((y / height) * (rows - 1))

    col_1 = min(col_ + 1, cols - 1)

    row_1 = min(row_ + 1, rows - 1)

    # q11 = nodes[row_, col_]

    # p00____p01
    # |       |
    # |       |
    # p10____p11

    dx_ = width / (cols - 1.0)
    dy_ = height / (rows - 1.0)

    tx = (x - dx_ * col_) / dx_
    ty = (y - dy_ * row_) / dy_
    pids = ((col_,  row_),  # p00
            (col_1, row_),  # p01
            (col_,  row_1),  # p10
            (col_1, row_1))  # p11

    b = np.zeros((16,), dtype=float)  # TODO CHECK IF np.zeros(...) MAY BE REPLACED BY SOMETHING

    c = np.zeros((16,), dtype=float)  # TODO CHECK IF np.zeros(...) MAY BE REPLACED BY SOMETHING

    for i in range(4):
        _col, _row = pids[i]
        b[i]      = points    [_row, _col]
        b[4 + i]  = points_dx [_row, _col]  # * dx_
        b[8 + i]  = points_dy [_row, _col]  # * dy_
        b[12 + i] = points_dxy[_row, _col]  # * dx_ * dy_

    c[1] = 1.0 * b[8]
    c[0] = 1.0 * b[0]
    c[2] = -3.0 * b[0] + 3.0 * b[2] - 2.0 * b[8] - 1.0 * b[10]
    c[3] = 2.0 * b[0] - 2.0 * b[2] + 1.0 * b[8] + 1.0 * b[10]
    c[4] = 1.0 * b[4]
    c[5] = 1.0 * b[12]
    c[6] = -3.0 * b[4] + 3.0 * b[6] - 2.0 * b[12] - 1.0 * b[14]
    c[7] = 2.0 * b[4] - 2.0 * b[6] + 1.0 * b[12] + 1.0 * b[14]
    c[8] = -3.0 * b[0] + 3.0 * b[1] - 2.0 * b[4] - 1.0 * b[5]
    c[9] = -3.0 * b[8] + 3.0 * b[9] - 2.0 * b[12] - 1.0 * b[13]
    c[10] = 9.0 * b[0] - 9.0 * b[1] - 9.0 * b[2] + 9.0 * b[3] + 6.0 * b[4] + 3.0 * b[5] - 6.0 * b[6] - 3.0 * b[
        7] + 6.0 * b[8] - 6.0 * b[9] + 3.0 * b[10] - 3.0 * b[11] + 4.0 * b[12] + 2.0 * b[13] + 2.0 * b[14] + 1.0 * b[15]
    c[11] = -6.0 * b[0] + 6.0 * b[1] + 6.0 * b[2] - 6.0 * b[3] - 4.0 * b[4] - 2.0 * b[5] + 4.0 * b[6] + 2.0 * b[
        7] - 3.0 * b[8] + 3.0 * b[9] - 3.0 * b[10] + 3.0 * b[11] - 2.0 * b[12] - 1.0 * b[13] - 2.0 * b[14] - 1.0 * b[15]
    c[12] = 2.0 * b[0] - 2.0 * b[1] + 1.0 * b[4] + 1.0 * b[5]
    c[13] = 2.0 * b[8] - 2.0 * b[9] + 1.0 * b[12] + 1.0 * b[13]
    c[14] = -6.0 * b[0] + 6.0 * b[1] + 6.0 * b[2] - 6.0 * b[3] - 3.0 * b[4] - 3.0 * b[5] + 3.0 * b[6] + 3.0 * b[
        7] - 4.0 * b[8] + 4.0 * b[9] - 2.0 * b[10] + 2.0 * b[11] - 2.0 * b[12] - 2.0 * b[13] - 1.0 * b[14] - 1.0 * b[15]
    c[15] = 4.0 * b[0] - 4.0 * b[1] - 4.0 * b[2] + 4.0 * b[3] + 2.0 * b[4] + 2.0 * b[5] - 2.0 * b[6] - 2.0 * b[
        7] + 2.0 * b[8] - 2.0 * b[9] + 2.0 * b[10] - 2.0 * b[11] + 1.0 * b[12] + 1.0 * b[13] + 1.0 * b[14] + 1.0 * b[15]

    return _cubic_poly(tx, ty, c)


@fast_math
def bi_qubic_interp_pt(x: float, y: float, points: np.ndarray, width: float = 1.0, height: float = 1.0) -> float:
    """
    Бикубическая интерполяция точки (x,y).
    :param x: x - координата точки.
    :param y: y - координата точки.
    :param points: одномерный список узловых точек.
    :param width: ширина области интерполяции.
    :param height: высота области интерполяции.
    :return:
    """
    if points.ndim != 2:
        raise RuntimeError("bi_linear_interp_pt :: points array has to be 2 dimensional")

    rows, cols = points.shape[0], points.shape[1]

    x = clamp(x, 0.0, width)

    y = clamp(y, 0.0, height)

    col_ = int((x / width) * (cols - 1))

    row_ = int((y / height) * (rows - 1))

    col_1 = min(col_ + 1, cols - 1)

    row_1 = min(row_ + 1, rows - 1)

    # q11 = nodes[row_, col_]

    # p00____p01
    # |       |
    # |       |
    # p10____p11

    dx_ = width / (cols - 1.0)
    dy_ = height / (rows - 1.0)

    tx = (x - dx_ * col_) / dx_
    ty = (y - dy_ * row_) / dy_
    pids = ((col_, row_),  # p00
            (col_1, row_),  # p01
            (col_, row_1),  # p10
            (col_1, row_1))  # p11

    b = np.zeros((16,), dtype=float)

    c = np.zeros((16,), dtype=float)

    for i in range(4):
        _col, _row = pids[i]
        b[i] = points[_row, _col]
        dx, dy, dxy = compute_derivatives_2_at_pt(points, _row, _col)
        b[4 + i] = dx
        b[8 + i] = dy
        b[12 + i] = dxy

    c[1] = 1.0 * b[8]
    c[0] = 1.0 * b[0]
    c[2] = -3.0 * b[0] + 3.0 * b[2] - 2.0 * b[8] - 1.0 * b[10]
    c[3] = 2.0 * b[0] - 2.0 * b[2] + 1.0 * b[8] + 1.0 * b[10]
    c[4] = 1.0 * b[4]
    c[5] = 1.0 * b[12]
    c[6] = -3.0 * b[4] + 3.0 * b[6] - 2.0 * b[12] - 1.0 * b[14]
    c[7] = 2.0 * b[4] - 2.0 * b[6] + 1.0 * b[12] + 1.0 * b[14]
    c[8] = -3.0 * b[0] + 3.0 * b[1] - 2.0 * b[4] - 1.0 * b[5]
    c[9] = -3.0 * b[8] + 3.0 * b[9] - 2.0 * b[12] - 1.0 * b[13]
    c[10] = 9.0 * b[0] - 9.0 * b[1] - 9.0 * b[2] + 9.0 * b[3] + 6.0 * b[4] + 3.0 * b[5] - 6.0 * b[6] - 3.0 * b[ 7]\
            + 6.0 * b[8] - 6.0 * b[9] + 3.0 * b[10] - 3.0 * b[11] + 4.0 * b[12] + 2.0 * b[13] + 2.0 * b[14] + 1.0 * b[15]
    c[11] = -6.0 * b[0] + 6.0 * b[1] + 6.0 * b[2] - 6.0 * b[3] - 4.0 * b[4] - 2.0 * b[5] + 4.0 * b[6] + 2.0 * b[7] \
            - 3.0 * b[8] + 3.0 * b[9] - 3.0 * b[10] + 3.0 * b[11] - 2.0 * b[12] - 1.0 * b[13] - 2.0 * b[14] - 1.0 * b[15]
    c[12] = 2.0 * b[0] - 2.0 * b[1] + 1.0 * b[4] + 1.0 * b[5]
    c[13] = 2.0 * b[8] - 2.0 * b[9] + 1.0 * b[12] + 1.0 * b[13]
    c[14] = -6.0 * b[0] + 6.0 * b[1] + 6.0 * b[2] - 6.0 * b[3] - 3.0 * b[4] - 3.0 * b[5] + 3.0 * b[6] + 3.0 * b[ 7] -\
            4.0 * b[8] + 4.0 * b[9] - 2.0 * b[10] + 2.0 * b[11] - 2.0 * b[12] - 2.0 * b[13] - 1.0 * b[14] - 1.0 * b[15]
    c[15] = 4.0 * b[0] - 4.0 * b[1] - 4.0 * b[2] + 4.0 * b[3] + 2.0 * b[4] + 2.0 * b[5] - 2.0 * b[6] - 2.0 * b[ 7] +\
            2.0 * b[8] - 2.0 * b[9] + 2.0 * b[10] - 2.0 * b[11] + 1.0 * b[12] + 1.0 * b[13] + 1.0 * b[14] + 1.0 * b[15]

    return _cubic_poly(tx, ty, c)


@fast_math
def bi_cubic_interp_derivatives_pt(x: float, y: float, points: np.ndarray, width: float = 1.0,
                                   height: float = 1.0, dx: float = 0.001, dy: float = 0.001) -> Tuple[float, float]:
    dx = width * dx
    dy = height * dy
    return (bi_qubic_interp_pt(x + dx, y, points, width, height) -
            bi_qubic_interp_pt(x - dx, y, points, width, height)) * 0.5 / dx, \
           (bi_qubic_interp_pt(x, y + dy, points, width, height) -
            bi_qubic_interp_pt(x, y - dy, points, width, height)) * 0.5 / dy


@fast_math
def bi_cubic_interp_derivatives2_pt(x: float, y: float, points: np.ndarray, width: float = 1.0,
                                    height: float = 1.0, dx: float = 0.001, dy: float = 0.001) -> \
        Tuple[float, float, float]:
    dx = width * dx
    dy = height * dy

    f_p_dx = bi_qubic_interp_pt(x + dx, y, points, width, height)
    f_m_dx = bi_qubic_interp_pt(x - dx, y, points, width, height)
    f_p_dy = bi_qubic_interp_pt(x, y + dy, points, width, height)
    f_m_dy = bi_qubic_interp_pt(x, y - dy, points, width, height)

    f_p_dy_dx = bi_qubic_interp_pt(x + dx, y + dy, points, width, height)
    f_m_dy_dx = bi_qubic_interp_pt(x - dx, y - dy, points, width, height)
    return (f_p_dx - f_m_dx) * 0.5 / dx, \
           (f_p_dy - f_m_dy) * 0.5 / dy, \
           (f_p_dy_dx - f_p_dx - f_p_dy + f_m_dy_dx) * 0.25 / dx / dy


@parallel
def bi_cubic_interp_derivatives(x: np.ndarray, y: np.ndarray, points: np.ndarray, width: float = 1.0,
                                height: float = 1.0, dx: float = 0.001, dy: float = 0.001) -> np.ndarray:
    """
    Бикубическая интерполяция частных производных диапазона точек x, y.
    :param x: x - координаты точек.
    :param y: y - координаты точек.
    :param points: одномерный список узловых точек.
    :param width: ширина области интерполяции.
    :param height: высота области интерполяции.
    :param dx:
    :param dy:
    :return:
    """
    result = np.zeros((y.size, x.size, 2), dtype=float)
    for i in parallel_range(result.size // 2):
        res_row_, res_col_ = divmod(i, x.size)
        _x, _y = bi_cubic_interp_derivatives_pt(x[res_col_], y[res_row_], points, width, height, dx, dy)
        result[res_row_, res_col_, 0] = _x
        result[res_row_, res_col_, 1] = _y
    return result


@parallel
def bi_cubic_interp_derivatives2(x: np.ndarray, y: np.ndarray, points: np.ndarray, width: float = 1.0,
                                 height: float = 1.0, dx: float = 0.001, dy: float = 0.001) -> np.ndarray:
    """
    Бикубическая интерполяция частных производных диапазона точек x, y
    :param x: x - координаты точек
    :param y: y - координаты точек
    :param points: одномерный список узловых точек
    :param width: ширина области интерполяции
    :param height: высота области интерполяции
    :param dx:
    :param dy:
    :return:
    """
    result = np.zeros((y.size, x.size, 3), dtype=float)

    for i in parallel_range(result.size // 3):
        res_row_, res_col_ = divmod(i, x.size)
        _x, _y, _z = bi_cubic_interp_derivatives2_pt(x[res_col_], y[res_row_], points, width, height, dx, dy)
        result[res_row_, res_col_, 0] = _x
        result[res_row_, res_col_, 1] = _y
        result[res_row_, res_col_, 2] = _z
    return result


@parallel
def bi_qubic_interp(x: np.ndarray, y: np.ndarray,
                    points: np.ndarray, width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """
    Би-кубическая интерполяция диапазона точек x, y.
    :param x: x - координаты точек.
    :param y: y - координаты точек.
    :param points: одномерный список узловых точек.
    :param width: ширина области интерполяции.
    :param height: высота области интерполяции.
    :return:
    """
    result = np.zeros((y.size, x.size), dtype=float)

    points_dx, points_dy, points_dxy = compute_derivatives_2(points)

    for i in parallel_range(result.size):
        res_row_, res_col_ = divmod(i, x.size)
        result[res_row_, res_col_] = _bi_qubic_interp_pt(x[res_col_], y[res_row_], points, points_dx,
                                                         points_dy, points_dxy, width, height)
    return result


@parallel
def bi_qubic_cut(x_0: float, y_0: float, x_1: float, y_1: float, steps_n: int, points: np.ndarray,
                 width: float = 1.0, height: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Сечение интерполируемой би-кубическим методом поверхности вдоль прямой через две точки.
    :param x_0: x - координата первой точки секущей.
    :param y_0: y - координата первой точки секущей.
    :param x_1: x - координата второй точки секущей.
    :param y_1: y - координата второй точки секущей.
    :param steps_n: количество точке вдоль секущей.
    :param points: одномерный список узловых точек.
    :param width: ширина области интерполяции.
    :param height: высота области интерполяции.
    :return:
    """
    dx = x_1 - x_0
    dy = y_1 - y_0
    rho = dx * dx + dy * dy
    if rho > 1e-12:
        rho = sqrt(rho)
        dx /= rho
        dy /= rho
    else:
        dx = 0.0
        dy = 0.0

    dt = 1.0 / (steps_n - 1)

    points_x = np.zeros((steps_n,), dtype=float)

    points_y = np.zeros((steps_n,), dtype=float)

    points_fxy = np.zeros((steps_n,), dtype=float)

    for i in parallel_range(steps_n):
        points_x[i] = dt * dx * i + x_0
        points_y[i] = dt * dy * i + y_0
        points_fxy[i] = bi_qubic_interp_pt(points_x[i], points_y[i], points, width, height)

    return points_x, points_y, points_fxy


@parallel
def bi_qubic_cut_along_curve(x_pts: np.ndarray, y_pts: np.ndarray, points: np.ndarray,
                             width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """
    Сечение интерполируемой би-кубическим методом поверхности вдоль кривой, заданной в виде массива точек.
    :param x_pts: координаты кривой по х.
    :param y_pts: координаты кривой по y.
    :param points: одномерный список узловых точек.
    :param width: ширина области интерполяции.
    :param height: высота области интерполяции.
    :return:
    """
    cut_values = np.zeros((min(x_pts.size, y_pts.size),), dtype=float)
    for i in parallel_range(cut_values.size):
        cut_values[i] = bi_qubic_interp_pt(x_pts[i], y_pts[i], points, width, height)
    return cut_values
