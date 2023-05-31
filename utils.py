import taichi as ti
from math import pi

vec2 = ti.types.vector(2, ti.f32)
vec3 = ti.types.vector(3, ti.f32)
vec4 = ti.types.vector(4, ti.f32)

mat2 = ti.math.mat2
mat3 = ti.math.mat3

@ti.func
def clamp(x, minVal, maxVal):
    """
    constrain a value to lie between two further values

    Parameters:
    x : Specify the value to constrain.

    minVal : Specify the lower end of the range into which to constrain x.

    maxVal : Specify the upper end of the range into which to constrain x.
    """
    return ti.min(ti.max(x, minVal), maxVal)

@ti.func
def smoothstep(edge0, edge1, x):
    """
    performs smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1

    Parameters:
    edge0 : Specifies the value of the lower edge of the Hermite function.

    edge1 : Specifies the value of the upper edge of the Hermite function.

    x : Specifies the source value for interpolation.
    """
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

@ti.func
def length(vec):
    """ vector or scalar length """
    return ti.sqrt(vec.dot(vec))

@ti.func
def sign(x: ti.f32):
    return 1. if x > 0. else -1. if x < 0. else 0.

@ti.func
def mix(x: ti.f32, y: ti.f32, a: ti.f32):
    '''
    performs a linear interpolation between x and y using a to weight between them. 
    The return value is computed as x * (1 - a) + y * a
    Parameters
    x : Specify the start of the range in which to interpolate.

    y : Specify the end of the range in which to interpolate.

    a : Specify the value to use to interpolate between x and y.
    '''
    return x * (1. - a) + y * a

@ti.func
def radians(angle):
    """ Positive angle to radians """
    return (angle - angle // 360) / 360. * 2. * pi

@ti.func
def rot(a):
    '''return rotation matrix'''
    return mat2(ti.cos(a), -ti.sin(a), 
                ti.sin(a), ti.cos(a))

@ti.func
def multiply2_right(mat, vec):
    '''multiply 2-d vector on (2, 2) matrix (from the right) result = M * v'''
    return vec2(vec.dot(mat[0, :]), vec.dot(mat[1, :]))

@ti.func
def dot(p, q):
    """ Scalar product of 2 vectors """
    return p.dot(q)

@ti.func
def sdf_lenses(p: vec2):

    half_side = 0.5
    q = abs(p) - vec2(half_side)

    distance = length(ti.max(q, 0.))

    center = vec2(half_side, 0.)
    radius = half_side
    distance += abs(min(length(p - center) - radius, 0.))

    return distance

@ti.func
def sdf_equilateral_triangle(p : vec2):
    k = ti.sqrt(3.0)
    r = 0.25
    p.x = ti.abs(p.x) - r
    p.y = p.y + r / k
    if p.x + k * p.y > 0.0:
        p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0
    p.x -= clamp( p.x, -2.0 * r, 0.0)
    return -length(p) * sign(p.y)

@ti.func
def get_rgb(matrix, row):
    return vec3(
        matrix[row, 0],
        matrix[row, 1],
        matrix[row, 2]
        )

