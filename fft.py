import numpy as np
import matplotlib.pyplot as plt

def triangle_wave(x):
    if 0.0 <= x <= np.pi:
        fx = -2.0/np.pi*x + 1.0  
    else :
        fx = 2.0*x/np.pi + 1.0
    return fx

def fs_tw(x,N):
    fx = 0.0
    for n in range(1, N+1):
        fx += (4*(1-np.power(-1,n)))/(np.power(np.pi,2)*np.power(n,2))*np.cos(float(n)*x)
    return fx

def fs_2(x,N):
    fx = 0.0
    for n in range(1,N+1):
        fx += (-2.0)/(np.power(np.pi,2))*np.cos(float(n)*x)
    return fx

def fs_3(x,N):
    fx = 0.0
    for n in range(1, 2*N+1, 2):
        fx += (8.0/(np.power(np.pi,2)*np.power(float(n),2)))*(np.power(-1.0,(n-1)/2))*np.sin(float(n)*x)
    return fx


delx = 0.05
plt.axis([-np.pi, np.pi,-2,2])
plt.xlabel('x co-ordinate')
plt.ylabel('Triangular_Wave(x)')

xr = np.arange(-np.pi, np.pi, delx)

twave = [triangle_wave(x) for x in xr] 

#absolutely brutal approximation:
y = np.cos(xr)
#N term Fourier series.
N = 10
#FS 1 (correct)
y1 = [fs_tw(x, N) for x in xr]
#plt.plot(xr,twave,'r', xr, y,'b--')
#FS 2
y2 = [fs_2(x, N) for x in xr]
#FS 3
y3 = [fs_3(x, N) for x in xr]
plt.plot(xr, twave,'r', xr, y,'b--', xr, y1, 'g--', xr, y2, 'c--', xr, y3, 'y--' )

plt.show()


