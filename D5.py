import math, cmath
import matplotlib.pyplot as plt

def koch_segment(A, B, depth):
    """递归生成从 A 到 B 的 Koch 曲线折线"""
    if depth == 0:
        return [A, B]
    v = B - A
    p1 = A + v/3
    p3 = A + 2*v/3
    # 在 p1-p3 上外凸 60° 形成顶点
    p2 = p1 + (v/3) * cmath.exp(-1j * math.pi/3)
    # 递归四段
    left  = koch_segment(A, p1, depth-1)[:-1]
    s1    = koch_segment(p1, p2, depth-1)[:-1]
    s2    = koch_segment(p2, p3, depth-1)[:-1]
    right = koch_segment(p3, B, depth-1)
    return left + s1 + s2 + right

def koch_snowflake(depth):
    A = 0+0j
    B = 1+0j
    C = 0.5 + 1j*math.sqrt(3)/2
    pts = (koch_segment(A, B, depth)[:-1] +
           koch_segment(B, C, depth)[:-1] +
           koch_segment(C, A, depth))
    return pts

# 示例绘制
depth = 10 
pts = koch_snowflake(depth)
xs = [z.real for z in pts] + [pts[0].real]
ys = [z.imag for z in pts] + [pts[0].imag]

plt.figure(figsize=(6,6))
plt.plot(xs, ys, '-', linewidth=1)
plt.axis('equal'); plt.axis('off')
plt.show()
