import matplotlib.pyplot as plt
import numpy as np


a1,a2,b1,b2 = -2.0467, 6.9670, -3.4223, 3.5
x3 = np.arange(a1, a2, 0.01)
y31 =(-(b2/a1)*x3 + b2)*(x3<=0)+ (-(b2/a2)*x3 + b2)*(x3>0)
y32 =(-(b1/a1)*x3 + b1)*(x3<=0)+ (-(b1/a2)*x3 + b1)*(x3>0)
fig, ax = plt.subplots()
ax.plot(x3, y31, x3, y32, color='royalblue', linewidth=2)
ax.fill_between(x3, y31, y32, where=(y32 <= y31), facecolor='deepskyblue', alpha=0.3,  label='LiveDeep')

a1,a2,b1,b2 = -4.6267, 5.0320, -4.9186, 5
x2 = np.arange(a1, a2, 0.01)
y21 =(-(b2/a1)*x2 + b2)*(x2<=0)+ (-(b2/a2)*x2 + b2)*(x2>0)
y22 =(-(b1/a1)*x2 + b1)*(x2<=0)+ (-(b1/a2)*x2 + b1)*(x2>0)
#fig, ax = plt.subplots()
ax.plot(x2, y21, x2, y22, color='green', linewidth=2)
ax.fill_between(x2, y21, y22, where=(y22 <= y21), facecolor='darkgreen', alpha=0.3, label='PanoSalNet')

a1,a2,b1,b2 = -7.4133, 2.9388, -7.2745, 7
x1 = np.arange(a1, a2, 0.01)
y11 =(-(b2/a1)*x1 + b2)*(x1<=0)+ (-(b2/a2)*x1 + b2)*(x1>0)
y12 =(-(b1/a1)*x1 + b1)*(x1<=0)+ (-(b1/a2)*x1 + b1)*(x1>0)
#fig, ax = plt.subplots()
ax.plot(x1, y11, x1, y12, color='orange', linewidth=4)
ax.fill_between(x1, y11, y12, where=(y12 <= y11), facecolor='lemonchiffon', alpha=0.8, label='GL360')

#横纵坐标
x = np.arange(-8, 8, 0.01)
y = np.arange(-8, 8, 0.01)
ax.plot(x,0*x, 0*y, y,color='black', linewidth=0.5)

#⚪
theta = np.linspace(0, 2 * np.pi, 200)

ax.plot(2* np.cos(theta), 2* np.sin(theta), color="gray", linewidth=1)
ax.plot(4* np.cos(theta), 4* np.sin(theta), color="gray", linewidth=1)
ax.plot(6* np.cos(theta), 6* np.sin(theta), color="gray", linewidth=1)
ax.plot(8* np.cos(theta), 8* np.sin(theta), color="gray", linewidth=1)

#字
plt.text(0,2, '10',size=10,color='black',weight=0)
plt.text(0,-2, '4',size=10,color='black',weight=0)
plt.text(0,4, '12',size=10,color='black',weight=0)
plt.text(0,-4, '6',size=10,color='black',weight=0)
plt.text(0,6, '14',size=10,color='black',weight=0)
plt.text(0,-6, '8',size=10,color='black',weight=0)
plt.text(0,8, '16',size=10,color='black',weight=0)
plt.text(0,-8, '10',size=10,color='black',weight=0)

plt.text(2,0, '5',size=10,color='black',weight=0)
plt.text(-2,0, '66',size=10,color='black',weight=0)
plt.text(4,0, '6',size=10,color='black',weight=0)
plt.text(-4,0, '69',size=10,color='black',weight=0)
plt.text(6,0, '7',size=10,color='black',weight=0)
plt.text(-6,0, '72',size=10,color='black',weight=0)
plt.text(8,0, '9',size=10,color='black',weight=0)
plt.text(-8,0, '75',size=10,color='black',weight=0)

plt.text(-9, -5.5, 'Average Bandwidth Saving [%]', size=12, color='black', weight=0, rotation=90)
plt.text(9.5, -7, 'Average Number of Prefetched Tiles [n]',size=12,color='black',weight=0, rotation=270)
plt.text(-4, -9, 'Average Utility Improvement',size=12,color='black',weight=0)
plt.text(-3.5, 9, 'Average Bitrate [Mbps]',size=12,color='black',weight=0)
#plt.yticks([-8, -4, 0, 4, 8],['$really\ bad$', '$bad$', '$normal$', '$good$', '$really\ good$'])

ax.xaxis.set_major_locator(plt.NullLocator()) # 删除x轴刻度，如需显示x轴刻度注释该行代码即可。
ax.yaxis.set_major_locator(plt.NullLocator()) # 删除y轴刻度，如需显示y轴刻度注释该行代码即可。

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
#ax.axis("equal")
ax.set_aspect(0.8)
#ax.set_title('Fill Between')
plt.xticks([])
plt.legend()

#figsize(12.5, 4) # 设置 figsize
plt.rcParams['savefig.dpi'] = 3000 #图片像素
plt.rcParams['figure.dpi'] = 3000


plt.savefig('plot123_2.png', dpi=120)
plt.show()