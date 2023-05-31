import taichi as ti
import taichi.math as tm
from utils import *
from mainImage import mainImage
import time

ti.init(arch=ti.gpu)
# ti.init(debug=True)

# resolution and pixels
asp = 16./9
h = 720
w = int(asp * h)
resolution = w, h
vec2_int = ti.types.vector(2, ti.i32)
iResolution = vec2_int(w, h)

acc = 0.1         # вес кадра для аккумулятора

imp_freq = 400.    # "частота" для генерации нескольких волн импульса
imp_sigma = vec2(1., asp) * 0.005
s_pos = vec2(-0.5, 0.)  # положение источника
angle = 0.


h = 1.0  # пространственный шаг решетки
c = 1.0  # скорость распространения волн
dt = h / (c * 1.5)  # временной шаг

n = vec3(  # коэффициент преломления
    1.30, # R
    1.35, # G
    1.40  # B
)


""" Declares a 1280x720 matrix field, each of its elements being a 3x3 matrix 
Once a field is declared, Taichi automatically initializes its elements to zero. """
tensor_field = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=resolution)

color = ti.Vector.field(3, dtype=ti.f32, shape=resolution)

accumulator = ti.Vector.field(3, dtype=ti.f32, shape=resolution)
""" аккумулятор для кадого цвета для накопления возмущений """

kappa = ti.Vector.field(3, dtype=ti.f32, shape=resolution)
""" вектор констант """

sdf_mask = ti.field(ti.f32, shape=resolution)
""" маска для линзы """

@ti.func
def define_mask(a: ti.f32 = 0.01, b: ti.f32 = 0.0):
    """
    Расчет треугольной маски размером (ny, nx) c плавным переходом между 0 и 1
    """
    w = iResolution.x
    h = iResolution.y
    for y in range(h):
        for x in range(w):
            uv = (vec2(x, y) - 0.5 * vec2(w, h)) / h
            # d = sdf_lenses(uv)
            d = sdf_equilateral_triangle(uv)
            sdf_mask[x, y] = smoothstep(a, b, d)

@ti.func
def wave_impulse(point, # vec2,
                 center_position, # vec2
                 freq: ti.f32,
                 sigma, # vec2
                ):
    """
    Импульс в виде нескольких сконцентрированных волн специальной формы для уменьшения расхождения "пучка".
    Форма - синусоида по направлению x под куполом функции Гаусса в направлениях x и y.
    https://graphtoy.com/?f1(x,t)=exp(-(x%5E2)/2.0/2%5E2)/2&v1=true&f2(x,t)=&v2=false&f3(x,t)=&v3=false&f4(x,t)=cos(20*x)*f1(x,t)&v4=true&f5(x,t)=&v5=false&f6(x,t)=&v6=false&grid=1&coords=0,0,4.205926793776742

    point : точка, в которой необходимо вычислить амплитуду импульса
    center_position : центр импульса
    freq : частота, отвечающая за количество возмущений внутри импульса
    sigma: размах купола Гаусса по осям x и y

    return : амплитуда импульса в точки point
    """
    d = (point - center_position) / sigma
    # d = (point - center_position) / vec2(0.001, asp)

    # (d[0]**2 / (sigma[0] ** 2) + d[1]**2 / (sigma[1] ** 2))
    return ti.exp(-0.05 * dot(d, d)) / 400. * ti.cos(freq * point.x)
    # return ti.exp(-0.5 * dot(d, d)) * ti.cos(freq * point.x)

@ti.func
def impulse():
    """
    Расчет импульса возмущений на нулевой итерации
    """
    s_alpha = -radians(angle)  # направление источника
    s_rot = rot(s_alpha)

    w = iResolution.x
    h = iResolution.y
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            uv = (vec2(x, y) - 0.5 * vec2(w, h)) / h
            init_impulse: ti.f32 = wave_impulse(multiply2_right(s_rot, uv), s_pos, imp_freq, imp_sigma)
            # init_impulse: ti.f32 = wave_impulse(uv, s_pos, imp_freq, imp_sigma)
            tensor_field[x, y] += init_impulse



@ti.func
def define_kappa():
    """ инициализация константы каппы """ 
    for fragCoord in ti.grouped(tensor_field):
        kappa[fragCoord] = (c * dt / h) * (sdf_mask[fragCoord] / n + 1. - sdf_mask[fragCoord])

@ti.func
def open_boundary():
    """
    Граничные условия открытой границы
    """
    w = iResolution.x
    h = iResolution.y

    for x in range(w):
        """ y = 0 """
        res = get_rgb(tensor_field[x, 1], 1) + (kappa[x, 0] - 1.) / (kappa[x, 0] + 1.) * (get_rgb(tensor_field[x, 1], 0) - get_rgb(tensor_field[x, 0], 1))
        tensor_field[x, 0][0, 0] = res.r
        tensor_field[x, 0][0, 1] = res.g
        tensor_field[x, 0][0, 2] = res.b
        """ y = h """
        res = get_rgb(tensor_field[x, h-2], 1) + (kappa[x, h-1] - 1.) / (kappa[x, h-1] + 1.) * (get_rgb(tensor_field[x, h-2], 0) - get_rgb(tensor_field[x, h-1], 1))
        tensor_field[x, h-1][0, 0] = res.r
        tensor_field[x, h-1][0, 1] = res.g
        tensor_field[x, h-1][0, 2] = res.b

    for y in range(h):
        """ x = 0 """
        res = get_rgb(tensor_field[1, y], 1) + (kappa[0, y] - 1.) / (kappa[0, y] + 1.) * (get_rgb(tensor_field[1, y], 0) - get_rgb(tensor_field[0, y], 1))
        tensor_field[0, y][0, 0] = res.r
        tensor_field[0, y][0, 1] = res.g
        tensor_field[0, y][0, 2] = res.b
        """ x = w """
        res  = get_rgb(tensor_field[w-2, y], 1) + (kappa[w-1, y] - 1.) / (kappa[w-1, y] + 1.) * (get_rgb(tensor_field[w-2, y], 0) - get_rgb(tensor_field[w-1, y], 1))
        tensor_field[w-1, y][0, 0] = res.r
        tensor_field[w-1, y][0, 1] = res.g
        tensor_field[w-1, y][0, 2] = res.b

@ti.func
def propagate():
    """
    Один шаг интегрирования уравнений распространения волны по Эйлеру
    """
    for fragCoord in ti.grouped(tensor_field):
        rgb = get_rgb(tensor_field[fragCoord], 1)
        tensor_field[fragCoord][2, 0] = rgb.r
        tensor_field[fragCoord][2, 1] = rgb.g
        tensor_field[fragCoord][2, 2] = rgb.b

        rgb = get_rgb(tensor_field[fragCoord], 0)
        tensor_field[fragCoord][1, 0] = rgb.r
        tensor_field[fragCoord][1, 1] = rgb.g
        tensor_field[fragCoord][1, 2] = rgb.b

    w = iResolution.x
    h = iResolution.y
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            for i in range(3):
                tensor_field[x, y][0, i] = kappa[x, y][i]**2 * (
                    tensor_field[x - 1, y][1, i] + tensor_field[x + 1, y][1, i] +
                    tensor_field[x, y - 1][1, i] + tensor_field[x, y + 1][1, i] -
                    4 * tensor_field[x, y][1, i]
                    ) + 2 * tensor_field[x, y][1, i] - tensor_field[x, y][2, i]

@ti.func
def accumulate():
    """
    Накопление возмущений, создаваемых волнами
    """
    w = iResolution.x
    h = iResolution.y
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            accumulator[x, y] += acc * ti.abs(get_rgb(tensor_field[x, y], 0)) * kappa[x, y] / (c * dt / h) / 2.

@ti.func
def create_rgb():
    for fragCoord in ti.grouped(accumulator):
        color[fragCoord] = clamp(accumulator[fragCoord], 0., 1.)


@ti.kernel
def init_kernel():
    """ с первой итерацией необходимо инициализировать каппу(константу) и придать начальный импульс системе """
    define_mask(0.01, 0.0)
    define_kappa()
    impulse()
    accumulate()
    
@ti.kernel
def render():
    open_boundary()
    propagate()
    accumulate()
    for fragCoord in ti.grouped(accumulator):
        color[fragCoord] = clamp(accumulator[fragCoord], 0., 1.)**2 + sdf_mask[fragCoord] * 0.2


# @ti.kernel
# def show_sdf():
#     define_mask()
#     for fragCoord in ti.grouped(accumulator):
#         color[fragCoord] = sdf_mask[fragCoord]

gui = ti.GUI("Wave", res=resolution, fast_gui=True)
frame = 0
start = time.time()

is_init_iteration = True

while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            break

    if is_init_iteration:
        init_kernel()
        is_init_iteration = False

    iTime = time.time() - start
    render()
    # show_sdf()
    gui.set_image(color)
    gui.show()
    frame += 1

gui.close()