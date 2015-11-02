import numpy as np
import matplotlib.pyplot as plt

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
plt.axis([-np.pi, np.pi,0,1.5])
bumps = [f_bumps(x) for x in xr]
bumps_fs = [fs_bumps(x, 5) for x in xr]
plt.plot(xr, bumps, 'r', xr, bumps_fs, 'go')

#Plucked String Fourier Series
plt.axis([-np.pi, np.pi,-1.5,1.5])
plucked = String()
string = [plucked.string(x) for x in xr]
fs_string = [plucked.fs_string(x,5) for x in xr]
fs_n = [plucked.fs_n(x/(np.pi*delx)) for x in xr]
plt.plot(xr, string, 'r', xr, fs_string, 'go', xr, fs_n, 'b-')
plt.show()

#Step Function and Sinc Function
plt.axis([-np.pi, np.pi,-1.5,2.5])
sqwave = SquareWave()
square = [sqwave.square(x) for x in xr]
fs_n   = [np.pi*sqwave.fs_n(x/(np.pi*delx)) for x in xr]
plt.plot(xr, square, 'r', xr, fs_n, 'b-')

plt.show()


