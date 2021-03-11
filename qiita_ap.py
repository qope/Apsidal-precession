import numpy as np
from gravipy.tensorial import *
from sympy import *
from itertools import product
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

t, r, theta, phi, M = symbols('t, r, \\theta, \phi, M')
chi = Coordinates('\chi', [t, r, theta, phi])
Metric = diag(-(1-2*M/r), 1/(1-2*M/r), r**2, r**2*sin(theta)**2)  #Schwarzschild計量
g = MetricTensor('g', chi, Metric)
Ga = Christoffel('Ga', g)
var("v_0, v_1, v_2, v_3")
var("a_0, a_1, a_2, a_3")
a_list = [a_0, a_1, a_2, a_3]
v_list = [v_0, v_1, v_2, v_3]
for i in range(4):
    a_list[i] = 0
for i, j, k in product(range(4), repeat=3):
    a_list[i] -= Ga( -i-1, j + 1, k + 1)*v_list[j]*v_list[k]

a_func = lambdify((t, r, theta, phi, M, v_0, v_1, v_2, v_3), a_list)
a = lambda x, v: np.array(a_func(x[0], x[1], x[2], x[3], 1, v[0], v[1], v[2], v[3]))

N = 10**6 #計算ステップ数
x = np.array([0.0, 17.32050808,  0.95531662, -0.78539816]) 
#初期位置は適当に決める。M=1がSchwarzschild半径なのでrはそれよりは大きくする
v = np.array([1, -0.02886728, -0.00824957,  0.01750001]) 
#t=0付近で\tau=tと選ぶとdt/d\tau = 1なので時間成分の速さは1にする
# 空間成分の速度は適当。

dtau = 0.01 #1ステップごとに進む\tau幅
R = [] 
Theta = []
Phi = []
T = []
for _ in range(N):
    T.append(x[0])
    R.append(x[1])
    Theta.append(x[2])
    Phi.append(x[3])
    k1v = a(x, v)*dtau
    k1x = v*dtau
    k2v = a(x+k1x/2, v+k1v/2)*dtau
    k2x = (v+k1v/2)*dtau
    k3v = a(x+k2x/2, v+k2v/2)*dtau
    k3x = (v+k2v/2)*dtau
    k4v = a(x+k3x, v+k3v)*dtau
    k4x = (v+k3v)*dtau
    v = v + (1/6)*(k1v+2*k2v+2*k3v+k4v)
    x = x + (1/6)*(k1x+2*k2x+2*k3x+k4x)
X = R*np.cos(Phi)*np.sin(Theta)
Y = R*np.sin(Phi)*np.sin(Theta)
Z = R*np.cos(Theta)

dt = 20 #時間幅
T_new = np.arange(0, T[-1], dt)
X_new = np.interp(T_new, T, X)
Y_new = np.interp(T_new, T, Y)
Z_new = np.interp(T_new, T, Z)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
L = 50
def update(i):
    if i != 0:
        ax.clear()
    ax.scatter(0, 0, 0, marker="o", c="orange", s=100)
    ax.plot(X_new[:i], Y_new[:i], Z_new[:i], c="black", alpha = 0.4)
    ax.scatter(X_new[i], Y_new[i], Z_new[i], marker="o", c="blue", s=10)
    ax.set_title(r"$t=$"+str(int(T_new[i])))
    ax.view_init(elev=30, azim=225)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)

ani = animation.FuncAnimation(fig, update, frames=len(T_new), interval=1)
plt.show()