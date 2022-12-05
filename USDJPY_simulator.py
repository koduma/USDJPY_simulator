from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import math
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import random
import matplotlib.animation as animation
import datetime

dayss = 7000

d1 = datetime.date(2004, 1, 27)

width = 24
height = 10000
dpp=[]

n=10000
peak_r=0.96
t=np.linspace(0,5200,n)

r1=np.random.choice([1,50],size=n,p=[peak_r,1-peak_r])
r=gaussian_filter1d(r1,20)
y=np.sin(t)*np.random.randn(n)*r
y/=250.0

def predict(S0,S,time):
    r=y[time]
    sigma = 0.00504#day
    T = 1#day
    t = width#hour
    dt = T / t
    M = height
    S = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt)* np.random.standard_normal((t + 1, M)), axis=0))
    P_call = sum(np.maximum(S[-1], 0)) / M
    #P_call=S0*(r+1.0)
    return P_call,S

price=105.62
price2=np.empty((width+1, height))
memo=np.empty((width+1, height))
dp = [0]*(dayss+100)
dp[0]=price

fig = plt3.figure()

ims = []

x_all=[]
y_all=[]

for i in range(dayss):
    result=predict(price,price2,i)
    price=result[0]
    price2=result[1]
    dp[i+1]=price
    x_all.append(d1 + datetime.timedelta(days=i))
    y_all.append(price)
    #im = plt3.plot(i,price,marker='o',color="k",markersize=5)
    im = plt3.plot(x_all,y_all, marker='o',color="red",markersize=2)
    ims.append(im)
    if i==0:
        memo=price2

ani = animation.ArtistAnimation(fig, ims, interval=100)
plt3.grid(True)
plt3.xlabel('date')
plt3.ylabel("U\nS\nD\nJ\nP\nY", labelpad=15, size=10,rotation=0,va='center')
plt3.ticklabel_format(style='plain',axis='y')
plt3.gca().yaxis.set_major_formatter(plt3.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))

plt.figure(figsize= (10,6))
plt.plot(memo[:,:100])
plt.grid(True)
plt.xlabel('hours')
plt.ylabel('Price')
plt.title('short span possible price paths')

plt2.figure(figsize= (10,6))
plt2.plot(x_all,y_all)
plt2.grid(True)
plt2.xlabel('date')
plt2.ylabel("U\nS\nD\nJ\nP\nY", labelpad=15, size=10,rotation=0,va='center')
plt2.title('long span price')
plt2.ticklabel_format(style='plain',axis='y')
plt2.gca().yaxis.set_major_formatter(plt2.FuncFormatter(lambda x, loc: '{:,}'.format(int(x))))

plt.show()
plt2.show()
plt3.show()

print(dp[dayss])
