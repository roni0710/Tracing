from Geometry import Vector3, tracing_2d_test, Transform3d
import matplotlib.pyplot as plt
from Geometry import Matrix3
from typing import Tuple
import numpy as np
import math
NUMERICAL_ACCURACY = 1e-6

class Sphere:
    __slots__ = ('radius', 'transform', 'b_param', 'd_param', 'k_param', 's_matrix', 'aperture_min', 'aperture_max')

    def __init__(self, radius: float, pos: Vector3 = None, rot: Vector3 = None):
        self.radius = float(radius)
        self.transform = Transform3d()
        self.s_matrix = Matrix3(1.0 / (self.radius * self.radius), 0, 0,
                                0, 1.0 / (self.radius * self.radius), 0,
                                0, 0, 1.0 / (self.radius * self.radius))
        self.b_param = Vector3(0, 0, 0)
        self.d_param = Vector3(0, 0, -self.radius)
        self.k_param = -1
        self.aperture_max = 2.0
        self.aperture_min = 0.0
        if pos:
            self.transform.origin = pos
        if rot:
            self.transform.angles = rot


# 1. не используй нампи для мтриц. у тебя сеть "from matrix3 import Matrix3"
# 2. используй конструкции вида: "if __name__ == "__main__":..." для обознаечния точки входа в программу
# 3. создай отдельную функцию для создания сферы. def make_sphere(r, pos):....
#    пусть для начала вернёт словарь со свойтвами: {"r": r, "transform": ... ,
#    "b_param": ..., "d_param": ..., "a_param": ..., 'k_param': ...}
# 4. Так же отдельную функцию для трейсинга def trace_ray(ray_origin, ray_direction, sphere_obj):
# 5. Построение поверхности - отдельная функция
# 6. Рисование поверхности - отдельная функция
# 7. Рисование списка лучей - отдельная функция
# 8. * Внутри "__main__" создать луч
#    * Создать сферу
#    * Cделать трасировку луча(пересечение со сферой)
#    * Создать геометрию для сферы
#    * Нарисовать сферу
#    * Нарисовать луч

def surface_normal(pos: Vector3, m_param: Matrix3, b_param: Vector3, d_param: Vector3) -> Vector3:
    """
   2M(pos-d) + b
   :param pos:
   :param m_param:
   :param b_param:
   :param d_param:
   :return:
   """
    return ((2.0 * Vector3.dot(m_param * pos, pos - d_param)) + b_param).normalize()


def intersect_surface(ro: Vector3, rd: Vector3,
                      m_param: Matrix3, b_param: Vector3,
                      d_param: Vector3, k_param: float) -> float:
    """
    Считает длину луча до пересечения с поверхностью
    :param ro: начало луча
    :param rd: единичный вектор направления луча
    :param m_param: матрица поверхности второго порятка
    :param b_param: линейный член поверхности второго порядка
    :param d_param: сдвиг по оси Z
    :param k_param:
    :return:
    """
    a_koef = Vector3.dot(m_param * rd, rd)
    b_koef = (2.0 * Vector3.dot(m_param * rd, ro) - 2.0 * Vector3.dot(m_param * rd, d_param) -
              Vector3.dot(m_param * d_param, rd) + Vector3.dot(b_param, rd))
    c_koef = (Vector3.dot(m_param * ro, ro) - 2.0 * Vector3.dot(m_param * d_param, ro) +
              Vector3.dot(m_param * d_param, d_param) -
              2.0 * Vector3.dot(b_param, ro) - 2.0 * Vector3.dot(b_param, d_param) + k_param)
    det = b_koef * b_koef - 4.0 * a_koef * c_koef
    if det < 0.0:
        return -1.0
    det = math.sqrt(det)
    t1 = (-b_koef + det) / (2.0 * a_koef)
    t2 = (-b_koef - det) / (2.0 * a_koef)
    if t1 < 0 and t2 < 0:
        return -1.0
    if t1 * t2 < 0:
        return max(t1, t2)
    return t2


def raytrace_surface(ro: Vector3, rd: Vector3,
                     m_param: Matrix3, b_param: Vector3,
                     d_param: Vector3, k_param: float) -> Tuple[float, Vector3, Vector3]:
    """
    :param ro:
    :param rd:
    :param m_param:
    :param b_param:
    :param d_param:
    :param k_param:
    :return: расстояние до точки пересечения, координаты точки пересечения и нормаль в точке пересечения
    """
    t = intersect_surface(ro, rd, m_param, b_param, d_param, k_param)
    if t < 0.0:
        return t, Vector3(), Vector3()
    t = intersect_surface(ro, rd, m_param, b_param, d_param, k_param)
    ray_end = rd * t + ro
    return t, ray_end, surface_normal(ray_end, m_param, b_param, d_param)


def trace_ray(ro: Vector3, rd: Vector3, surf: Sphere) -> Tuple[float, Vector3, Vector3]:
    """
    _rd = transform.inv_transform_vect(rd, 0.0)
    _ro = transform.inv_transform_vect(ro, 1.0)
    t, re, rn = _trace_surface_3d(_rd, _ro, radius)
    return (0.0, ro, re) if t < 0 else (t, transform.transform_vect(re), transform.transform_vect(rn, 0.0))

    :param ro:
    :param rd:
    :param surf:
    :return:
    """
    ro_ = surf.transform.inv_transform_vect(ro, 1.0)
    rd_ = surf.transform.inv_transform_vect(rd, 0.0)
    t, re, rn = raytrace_surface(ro_, rd_, surf.s_matrix, surf.b_param, surf.d_param, surf.k_param)
    return  (0.0, ro, re) if t < 0 else (t, surf.transform.transform_vect(re), surf.transform.transform_vect(rn, 0.0))


def surf_sag(x: float, y: float, m_surf: Matrix3, b_surf: Vector3,
             k_surf: float, s_orientation: float) -> Vector3:
    a = m_surf.m22
    b = b_surf.z + 2 * x * m_surf.m02 + 2 * y * m_surf.m12
    c = x * x * m_surf.m00 + 2 * x * y * m_surf.m01 + y * y * m_surf.m11 + x * b_surf.x + y * b_surf.y + k_surf
    if abs(a) < NUMERICAL_ACCURACY:
        return Vector3(x, y, -c / b) if abs(b) > NUMERICAL_ACCURACY else Vector3(x, y, 0.0)
    det = b * b - 4 * a * c
    if det < 0:
        return Vector3(x, y, 0.0)
    det = math.sqrt(det)
    t1, t2 = (-b + det) / (2 * a), (-b - det) / (2 * a)
    if t1 < 0 and t2 < 0:
        return Vector3(x, y, 0.0)
    if t1 * t2 < 0:
        return Vector3(x, y, (max(t1, t2) if s_orientation < 0.0 else min(t1, t2)))
    return Vector3(x, y, t2)


def surf_shape(surf: Sphere, steps_r: int = 32, steps_angle: int = 32):
    angles = np.linspace(0, np.pi * 2.0, steps_angle)
    radius = surf.aperture_min + np.sqrt(np.linspace(0.0, 1.0, steps_r)) * (surf.aperture_max - surf.aperture_min)
    xs = []
    ys = []
    zs = []
    for ri in radius.flat:
        x_row = []
        y_row = []
        z_row = []
        for ai in angles.flat:
            xi, yi, zi = surf_sag(ri * math.cos(ai), ri * math.sin(ai),
                                   surf.s_matrix, surf.b_param, surf.k_param, 1.0)
            x_row.append(xi)
            y_row.append(yi)
            z_row.append(zi)
        xs.append(x_row)
        ys.append(y_row)
        zs.append(z_row)
    return np.array(xs), np.array(ys), np.array(zs)


def draw_surf(surf: Sphere, axis=None) -> None:
        axis = axis if axis else plt.axes(projection='3d')
        surf = surf_shape(surf)
        axis.plot_surface(*surf, linewidths=0.0, antialiased=True, color='blue', edgecolor="none", alpha=0.6)
        axis.set_aspect('equal', 'box')
        axis.set_xlabel("z, [mm]")
        axis.set_ylabel("x, [mm]")
        axis.set_zlabel("y, [mm]")


def draw_ray(ro: Vector3, rd: Vector3, t: float, axis=None) -> None:
    x, y, z = ro
    x1, y1, z1 = ro + rd * t
    axis = axis if axis else plt.axes(projection='3d')
    axis.plot((x, x1), (y, y1), (z, z1))


if __name__ == "__main__":
    surf = Sphere(-16)
    ro = Vector3(1.0, 1.0, 1.0)
    rd = Vector3(1.0, 1.0, 1.0)
    t, re, rn = trace_ray(ro, rd, surf)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_surf(surf, ax)
    draw_ray(ro, rd, t, ax)
    plt.show()

    exit()
    tracing_2d_test()
    visualize_rays(ray_list, sphere_obj)