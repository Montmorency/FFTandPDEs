from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time

def triangle_wave(x):
    if 0.0 <= x <= np.pi:
        fx = -2.0/np.pi*x + 1.0  
    else :
        fx =  2.0*x/np.pi + 1.0
    return fx

# For Triangular wave we would find Fourier series:
# f(x) = \sum_{n=0}^{N} \frac{4(1 - (-1)^{n})}{\pi^{2} n^{2}}\cos(n*x)
def fs_tw(x,N):
    fx = 0.0
    for n in range(1, N+1):
        fx += (4*(1-np.power(-1,n)))/(np.power(np.pi,2)*np.power(n,2))*np.cos(float(n)*x)
    return fx

# Solution if wave is shifted right by  $\pi/2$:
# f(x) = \sum_{n=1,3,5,...}^{N}\frac{8}{\pi^{2} n^{2}} -1^{(n-1)/2}\sin(nx)
def fs_2(x,N):
    fx = 0.0
    for n in range(1, 2*N+1, 2):
        fx += (8.0/(np.power(np.pi,2)*np.power(float(n),2)))*(np.power(-1.0,(n-1)/2))*np.sin(float(n)*x)
    return fx

def f_bumps(x):
#1-\frac{4}{L^2}(x - \frac{L}{2})^{2}
    if x<0.0:
        x=x+np.pi
    fx = 0.0
    fx = 1.0 - (4.0/np.power(np.pi,2))*np.power((x-np.pi/2.0), 2)
    return fx

def fs_bumps(x,N):
    fx = (2.0/3.0)
    for n in range(1, N+1):
        fx += -(4.0/np.power(np.pi,2))*np.cos(2.0*float(n)*np.pi*x/np.pi)/np.power(float(n),2)
    return fx

class String(object):
    #Few method for 
    def __init__(self):
        self.d = np.pi*0.75
        self.delx = 0.05
        self.xr = np.arange(-np.pi,np.pi,self.delx)
        self.c = 1.0

    def string(self, x):
        a = 1.0
        if x < 0:
            x = - x
            a = -1.0
        if x < self.d:
            return a*x/self.d
        elif self.d <= x < np.pi:
            return a*(np.pi-x)/(np.pi - self.d)

    def fs_string(self, x, N):
        fx = 0.0
        for n in range(1,N+1):
            fx += 2*np.sin(float(n)*self.d)/(np.power(float(n),2)*self.d*(np.pi-self.d))*np.sin(float(n)*x)
        return fx

    def fs_n(self, n):
        #pass x in range of 0:np.pi
        #but wants to have integers for fs
        if n == 0:
            return 0.0
        else:
            return (2*np.sin(float(n)*self.d))/(np.power(n,2)*self.d*(np.pi-self.d))

    def wave(self, x, N, t):
        fx = 0.0
        for n in range(1, N+1):
            fx += self.fs_n(n)*np.sin(float(n)*x)*np.cos(self.c*n*t)
        return fx

    def animation(self):
	tt    = np.arange(0.0, 80.0, 0.1)
        pluck = [self.string(x) for x in self.xr]
    	plt.plot(self.xr, pluck,'r')
        plt.ion()
        plt.draw()
        for t in tt:
            y = [self.wave(x, 12, t) for x in self.xr]
    	    #plt.plot(self.xr, pluck, 'r', self.xr, y,'b--')
            plt.axis([0.0, np.pi,-1.5,1.5])
    	    plt.plot(self.xr, y,'b--')
    	    plt.draw()
            plt.clf()

class SquareWave(object):
    def __init__(self):
        self.L = 3

    def square(self, x):
        if (-1.0 < x < 1.0):
            return 1.0
        else:
            return 0.0

    def fs_n(self, x):
        if x != 0.0:
           return (np.sin(x)/(np.pi*x))
        else:
            return 1.0

class GasDiffusion(object):
    def __init__(self):
        self.L    = 20
        self.D    = 0.06
        self.delx = 0.1
	self.xr   = np.arange(0.0, self.L, delx)

    def poison(self, x):
        if 0.0 <= x <= self.L:
            fx = x/self.L   
        return fx

    def initial_dist(self):
	self.gas = [self.poison(x) for x in self.xr]
        return

    def fs_diffusion(self, x, N, t):
        fx = 0.0
    	for n in range(1, N+1, 2):
            fx += -4.0/(np.power(np.pi*float(n),2))*np.cos(np.pi*float(n)*x/self.L)*np.exp(-t*self.D*np.power(float(n)*np.pi/self.L,2))
    	    fx +=  0.5
    	    return fx

    def animation(self):
	y = [self.fs_diffusion(x, 5, 0.0) for x in self.xr]
	plt.plot(self.xr, self.gas,'r', self.xr, y,'b--')
	tt = np.arange(0.0,1500.0,10)
        plt.ion()
	plt.show()
	for t in tt:
            y = [self.fs_diffusion(x, 5, t) for x in self.xr]
    	    plt.plot(self.xr, self.gas,'r',self.xr, y,'b--')
    	    plt.draw()
    	    #time.sleep(0.25)


def sinxy(x,y,n,m):
    a = 2.0
    b = 1.0 
    fx = np.sin(float(n)*x*np.pi/a)*np.sin(float(m)*y*np.pi/b)
    return fx 

#def fs_2(x,N):
#    fx = 0.0
#    for n in range(1,N+1):
#        fx += (-2.0)/(np.power(np.pi,2))*np.cos(float(n)*x)
#    return fx
#Incorrect solution for triangular wave:
#FS 3
#y3 = [fs_3(x, N) for x in xr]

delx = 0.05
plt.axis([-np.pi, np.pi,-2,2])
plt.xlabel('x co-ordinate')
plt.ylabel('Triangular_Wave(x)')
xr = np.arange(-np.pi, np.pi, delx)
twave = [triangle_wave(x) for x in xr] 

#N term Fourier series.
y = np.cos(xr)
N = 10

#FS 1 (correct)
y1 = [fs_tw(x, N) for x in xr]
y2 = [fs_2(x, N) for x in xr]
#plt.plot(xr, twave,'r', xr, y,'b--', xr, y1, 'g--', xr, y2, 'y--')

#Bumps Fourier Series
#plt.axis([-np.pi, np.pi,0,1.5])
#bumps = [f_bumps(x) for x in xr]
#bumps_fs = [fs_bumps(x, 5) for x in xr]
#plt.plot(xr, bumps, 'r', xr, bumps_fs, 'go')

#Plucked String Fourier Series
#plt.axis([-np.pi, np.pi,-1.5,1.5])
#plucked = String()
#string = [plucked.string(x) for x in xr]
#fs_string = [plucked.fs_string(x,5) for x in xr]
#fs_n = [plucked.fs_n(x/(np.pi*delx)) for x in xr]
#plt.plot(xr, string, 'r', xr, fs_string, 'go', xr, fs_n, 'b-')
#plt.show()


#Step Function and Sinc Function
#plt.axis([-np.pi, np.pi,-1.5,2.5])
#sqwave = SquareWave()
#square = [sqwave.square(x) for x in xr]
#fs_n   = [np.pi*sqwave.fs_n(x/(np.pi*delx)) for x in xr]
#plt.plot(xr, square, 'r', xr, fs_n, 'b-')
#plt.show()

#Diffusion of Gas along a line
#pde = GasDiffusion()
#pde.initial_dist()
#pde.animation()

#Plucked String Fourier Series
plt.axis([-np.pi, np.pi, -1.5, 1.5])
plucked = String()
N = 10
#For D=0.75*pi
string = [plucked.string(x) for x in xr]
fs_string = [plucked.fs_string(x,N) for x in xr]
fs_n = [plucked.fs_n(x/(2*np.pi*np.pi*delx)) for x in xr]
xn = [float(n) for n in range(-4,4)]
fs_np = [plucked.fs_n(float(n)) for n in range(-4,4)]
plt.plot(xr, string, 'r', xr, fs_string, 'bx', xn, fs_np, 'bo', xr, fs_n, 'b-')
#plt.show()

#For D=0.51*pi
triangle = String()
plt.axis([-np.pi, np.pi,-1.5,1.5])
triangle.d = 0.51*np.pi
string = [triangle.string(x) for x in xr]
fs_string = [triangle.fs_string(x,N) for x in xr]
fs_n = [triangle.fs_n(x/(2*np.pi*np.pi*delx)) for x in xr]
xn = [float(n) for n in range(-4,4)]
fs_np = [triangle.fs_n(float(n)) for n in range(-4,4)]
plt.plot(xr, string, 'r', xr, fs_string, 'gx', xn, fs_np, 'go', xr, fs_n, 'g-')
plt.show()

#Animation for Different String Configurations
#plucked.animation()
plucked.d = 0.51*np.pi
#plucked.animation()
#Harmonic Modes of a 2d Rectangle.

X, Y = np.meshgrid(np.linspace(0,2,101),np.linspace(0,1,101))
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[0]):
        Z[i,j] = sinxy(X[i,j], Y[i,j], 3, 1)

fig  = plt.figure()
fig1 = plt.figure()
fig2 = plt.figure()

ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='x', offset=0, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='y', offset=1, cmap=cm.coolwarm)

for i in range(X.shape[0]):
    for j in range(X.shape[0]):
        Z[i,j] = sinxy(X[i,j], Y[i,j], 1, 1)

bx = fig1.gca(projection='3d')
bx.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = bx.contourf(X, Y, Z, zdir='z', offset=-2, cmap=cm.coolwarm)
cset = bx.contourf(X, Y, Z, zdir='x', offset=0, cmap=cm.coolwarm)
cset = bx.contourf(X, Y, Z, zdir='y', offset=1, cmap=cm.coolwarm)

for i in range(X.shape[0]):
    for j in range(X.shape[0]):
        Z[i,j] = sinxy(X[i,j], Y[i,j], 2, 1)

cx = fig2.gca(projection='3d')
cx.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = cx.contourf(X, Y, Z, zdir='z', offset=-2, cmap=cm.coolwarm)
cset = cx.contourf(X, Y, Z, zdir='x', offset=0, cmap=cm.coolwarm)
cset = cx.contourf(X, Y, Z, zdir='y', offset=1, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(0, 2)
ax.set_ylabel('Y')
ax.set_ylim(0, 2)
ax.set_zlabel('Z')
ax.set_zlim(-2, 2)

bx.set_xlabel('X')
bx.set_xlim(0, 2)
bx.set_ylabel('Y')
bx.set_ylim(0, 2)
bx.set_zlabel('Z')
bx.set_zlim(-2, 2)

cx.set_xlabel('X')
cx.set_xlim(0, 2)
cx.set_ylabel('Y')
cx.set_ylim(0, 2)
cx.set_zlabel('Z')
cx.set_zlim(-2, 2)

plt.show()

