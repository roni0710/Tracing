from Geometry import Vector3, tracing_2d_test, Transform3d, Vector2
from Geometry import Matrix3, NUMERICAL_ACCURACY
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import math

SPHERICAL_SURFACE = 0
ELLIPSOIDAL_SURFACE = 1
HYPERBOLIC_SURFACE = 2
PARABOLOID_SURFACE = 3
ELLIPTIC_PARABOLOID_SURFACE = 4
HYPERBOLIC_PARABOLOID_SURFACE = 5
CONIC_SURFACE = 6
USER_DEFINED_SURFACE = 7

MATERIAL = 'material'
GLASS = 'glass'
MIRROR = 'mirror'
GLASS_PARAMS = 'glass_params'
IMAGE_OBJECT = 'image'

class SecondOrderSurface:
    __slots__ = ('_transform', '_b_param', '_d_param', '_k_param',
                 '_s_matrix', '_aperture', '_direction', '_surface_type', '_material')

    def __init__(self, axis:  Vector3, pos: Vector3 = None, rot: Vector3 = None):
        self._transform = Transform3d()
        self._direction = 1.0 if axis.z > 0.0 else -1.0
        self._s_matrix = Matrix3(1.0 / (axis.x * axis.x), 0, 0,
                                0, 1.0 / (axis.y * axis.y), 0,
                                0, 0, 1.0 / (axis.z * axis.z))
        self._b_param = Vector3(0, 0, 0)
        self._d_param = Vector3(0, 0, -axis.z)
        self._k_param = -1.0
        self._aperture = Vector2(0.0, min(abs(axis.x), abs(axis.y)))
        self._material = {MATERIAL: GLASS, GLASS_PARAMS: (1.0, 1.33)}
        if pos:
            self.transform.origin = pos
        if rot:
            self.transform.angles = rot

    @classmethod
    def build_ellipsoid(cls, semi_axis: Vector3, pos: Vector3 = None, rot: Vector3 = None) -> 'SecondOrderSurface':
        result = cls(semi_axis, pos, rot)
        result._surface_type = ELLIPSOIDAL_SURFACE
        return result

    @classmethod
    def build_sphere(cls, radius: float, pos: Vector3 = None, rot: Vector3 = None) -> 'SecondOrderSurface':
        result = cls(Vector3(radius, radius, radius), pos, rot)
        result._surface_type = SPHERICAL_SURFACE
        return result

    @classmethod
    def build_paraboloid(cls, surface_args: Vector3, pos: Vector3 = None, rot: Vector3 = None) -> 'SecondOrderSurface':
        result = cls(surface_args, pos, rot)
        result._surface_type = PARABOLOID_SURFACE
        result._s_matrix.m22 = 0.0
        result._b_param.z = surface_args.z
        result._d_param.z = 0.0
        result._k_param = 0.0
        return result

    @classmethod
    def make_conic(cls, surface_args: Vector3, pos: Vector3 = None, rot: Vector3 = None) -> 'SecondOrderSurface':
        result = cls(surface_args, pos, rot)
        result._surface_type = CONIC_SURFACE
        result._s_matrix.m22 = -surface_args.z * surface_args.z
        result._direction *= -1.0
        result._d_param.z = 0.0
        result._k_param = 0.0
        return result

    @classmethod
    def make_image(cls, pos: Vector3 = None, rot: Vector3 = None) -> 'SecondOrderSurface':
        result = cls(Vector3(1e2, 1e2, 1e2), pos, rot)
        result._surface_type = IMAGE_OBJECT
        result.aperture_max = 1.0
        return result

    @property
    def material(self) -> dict:
        return self._material

    @property
    def transform(self) -> Transform3d:
        return self._transform

    @property
    def k_param(self) -> float:
        return self._k_param

    @property
    def direction(self) -> float:
        return self._direction

    @property
    def surface_type(self) -> int:
        return self._surface_type

    @property
    def d_param(self) -> Vector3:
        return self._d_param

    @property
    def b_param(self) -> Vector3:
        return self._b_param

    @property
    def s_matrix(self) -> Matrix3:
        return self._s_matrix

    @property
    def aperture_max(self) -> float:
        return self._aperture.y

    @property
    def aperture_min(self) -> float:
        return self._aperture.x

    @aperture_max.setter
    def aperture_max(self, value: float) -> None:
        self._aperture.y = max(self.aperture_min, float(value))

    @aperture_min.setter
    def aperture_min(self, value: float) -> None:
        self._aperture.x = min(self.aperture_max, float(value))


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

def surface_normal(pos: Vector3, m_param: Matrix3, b_param: Vector3) -> Vector3:  # , d_param: Vector3) -> Vector3:
    """
   2M(pos-d) + b
   :param pos:
   :param m_param:
   :param b_param:
   :param d_param:
   :return:
   """
    # return (2.0 * m_param * (pos - d_param) + b_param).normalize()
    return (2.0 * m_param * pos + b_param).normalize()


def intersect_surface(ro: Vector3, rd: Vector3,
                      m_param: Matrix3, b_param: Vector3,
                      k_param: float, direction: float) -> float:
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
    # b_koef = (2.0 * Vector3.dot(m_param * rd, ro) - 2.0 * Vector3.dot(m_param * rd, d_param) -
    #           Vector3.dot(m_param * d_param, rd) + Vector3.dot(b_param, rd))
    b_koef = (Vector3.dot(m_param * rd, ro) + Vector3.dot(m_param * ro, rd) + Vector3.dot(b_param, rd))
    # c_koef = (Vector3.dot(m_param * ro, ro) - 2.0 * Vector3.dot(m_param * d_param, ro) +
    #           Vector3.dot(m_param * d_param, d_param) -
    #           2.0 * Vector3.dot(b_param, ro) - 2.0 * Vector3.dot(b_param, d_param) + k_param)
    c_koef = (Vector3.dot(m_param * ro, ro) + Vector3.dot(b_param, ro) + k_param)

    if abs(a_koef) < NUMERICAL_ACCURACY:
        return -c_koef / b_koef
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
    return max(t1, t2) if direction < 0.0 else min(t1,t2)

def raytrace_surface(ro: Vector3, rd: Vector3,
                     m_param: Matrix3, b_param: Vector3, k_param: float, direction: float) -> (
        Tuple)[float, Vector3, Vector3]:
    """
    :param ro:
    :param rd:
    :param m_param:
    :param b_param:
    :param d_param:
    :param k_param:
    :return: расстояние до точки пересечения, координаты точки пересечения и нормаль в точке пересечения
    """
    t = intersect_surface(ro, rd, m_param, b_param, k_param, direction)
    if t < 0.0:
        return -1.0, Vector3(0, 0, 0), Vector3(0, 0, 0)
    ray_end = rd * t + ro
    return t, ray_end, surface_normal(ray_end, m_param, b_param)


def trace_ray(surf: SecondOrderSurface, ro: Vector3, rd: Vector3) -> Tuple[float, Vector3, Vector3]:
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
    ro_ = surf.transform.inv_transform_vect(ro, 1.0) + surf.d_param
    rd_ = surf.transform.inv_transform_vect(rd, 0.0)
    t, re, rn = raytrace_surface(ro_, rd_, surf.s_matrix, surf.b_param, surf.k_param, surf.direction)
    return (-1.0, ro, rd) if t < 0 else (t, surf.transform.transform_vect(re - surf.d_param),
                                         surf.transform.transform_vect(rn, 0.0))


def reflect_ray(surf: SecondOrderSurface, ro: Vector3, rd: Vector3) -> Tuple[float, Vector3, Vector3]:
    t, re, rn = trace_ray(surf, ro, rd)
    return (-1.0, ro, rd) if t < 0 else (t, re, Vector3.reflect(rd, rn))


def refract_ray(surf: SecondOrderSurface, ro: Vector3, rd: Vector3) -> Tuple[float, Vector3, Vector3]:
    t, re, rn = trace_ray(surf, ro, rd)
    ri1, ri2 = surf.material[GLASS_PARAMS]
    return (-1.0, ro, rd) if t < 0 else (t, re, Vector3.refract(rd, rn, ri1, ri2))


raytrace_actions = {IMAGE_OBJECT: trace_ray,
                    MIRROR: reflect_ray,
                    GLASS: refract_ray}


def ray_trace_surface(surf, ro: Vector3, rd: Vector3) -> Tuple[float, Vector3, Vector3]:
    try:
        m_type = surf._material[MATERIAL]
        return raytrace_actions[m_type](surf, ro, rd)
    except KeyError as er:
        print(f"refract_ray KeyError :: {er}")
        return 0.0, ro, rd
    except ValueError as er:
        print(f"refract_ray ValueError :: {er}")
        return 0.0, ro, rd


def surf_sag(x: float, y: float, m_surf: Matrix3, b_surf: Vector3, d_surf: Vector3,
             k_surf: float, s_orientation: float) -> Vector3:
    a = m_surf.m22
    b = b_surf.z + 2 * x * m_surf.m02 + 2 * y * m_surf.m12
    c = x * x * m_surf.m00 + 2 * x * y * m_surf.m01 + y * y * m_surf.m11 + x * b_surf.x + y * b_surf.y + k_surf
    if abs(a) < NUMERICAL_ACCURACY:
        return (Vector3(x, y, -c / b) if abs(b) > NUMERICAL_ACCURACY else Vector3(x, y, 0.0)) - d_surf
    det = b * b - 4 * a * c
    if det < 0:
        return Vector3(x, y, 0.0)
    det = math.sqrt(det)
    t1, t2 = (-b + det) / (2 * a), (-b - det) / (2 * a)
    if t1 < 0 and t2 < 0:
        return Vector3(x, y, 0.0)
    if t1 * t2 < 0:
        return Vector3(x, y, (max(t1, t2) if s_orientation < 0.0 else min(t1, t2))) - d_surf
    return Vector3(x, y, t2) - d_surf


def surf_shape(surf: SecondOrderSurface, steps_r: int = 32, steps_angle: int = 32):
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
            position = surf_sag(ri * math.cos(ai), ri * math.sin(ai),
                                  surf.s_matrix, surf.b_param, surf.d_param, surf.k_param,  surf.direction)
            xi, yi, zi = surf.transform.transform_vect(position, 1.0)
            x_row.append(xi)
            y_row.append(yi)
            z_row.append(zi)
        xs.append(x_row)
        ys.append(y_row)
        zs.append(z_row)
    return np.array(xs), np.array(ys), np.array(zs)


def draw_surf(surf: SecondOrderSurface, axis=None) -> None:
        axis = axis if axis else plt.axes(projection='3d')
        surf = surf_shape(surf)
        axis.plot_surface(*surf, linewidths=0.0, antialiased=True, color='blue', edgecolor="none", alpha=0.8)
        # axis.set_aspect('equal', 'box')
        axis.set_xlabel("z, [mm]")
        axis.set_ylabel("x, [mm]")
        axis.set_zlabel("y, [mm]")


def draw_ray(ro: Vector3, rd: Vector3, t: float, axis=None) -> None:
    axis = axis if axis else plt.axes(projection='3d')
    x, y, z = ro
    x1, y1, z1 = ro + rd * t
    axis.plot((x, x1), (y, y1), (z, z1), 'b')


def lens_tracing_test():
    surf1 = SecondOrderSurface.build_sphere(10)
    surf1.aperture_max = 2.5
    surf2 = SecondOrderSurface.build_sphere(-10)
    surf2.transform.origin = Vector3(0, 0, 1)
    surf1.material[GLASS_PARAMS] = (1.0, 1.333)
    surf2.material[GLASS_PARAMS] = (1.333, 1.0)
    surfaces = [surf1, surf2, SecondOrderSurface.make_image(pos=Vector3(0, 0, 5))]
    surf2.aperture_max = 2.5
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal', 'box')
    [draw_surf(surf, ax) for surf in surfaces]
    positions = []
    n_rays = 16
    for index in range(n_rays):
        x = index / (n_rays - 1.0)
        origin = Vector3(x, 0, -5)
        direction = Vector3(0, 0, 1)
        ray_path = []
        for surf in surfaces:
            ray_path.append(origin)
            t, origin, direction = ray_trace_surface(surf, origin, direction)
        ray_path.append(origin)
        positions.append(ray_path)
    for ray_path in positions:
        x = [v.x for v in ray_path]
        y = [v.y for v in ray_path]
        z = [v.z for v in ray_path]
        ax.plot(x, y, z, 'b')
        ax.plot(x, y, z, '.r')
    plt.show()



if __name__ == "__main__":
    lens_tracing_test()
    exit(1)
    surf = SecondOrderSurface.build_sphere(-5)
    surf.aperture_min = 0
    surf.aperture_max = 2.5
    surf.transform.origin = Vector3(0, 0, 0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal', 'box')
    draw_surf(surf, ax)
    di = 2 / 50
    for i in range(10):
        ro = Vector3(di * i, di * i, -5.0)
        rd = Vector3(0.0, 0.0, 1.0)
        t, re, rn = trace_ray(ro, rd, surf)
        draw_ray(ro, rd, t, ax)

    plt.show()

    exit()
    tracing_2d_test()
    visualize_rays(ray_list, sphere_obj)