import numpy as np
import matplotlib.pyplot as plt
import matplotlib

config = {
    "font.family":'serif',
    "font.size": 15,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
    "axes.unicode_minus": False, # 解决负号无法显示的问题
}
matplotlib.rcParams.update(config)

# 定义参数
epsilon_start = 1.0
epsilon_finish = 0.05
epsilon_anneal_time = 100000
delta = (epsilon_start - epsilon_finish) / epsilon_anneal_time



# 生成 t 值
t = np.arange(0, epsilon_anneal_time * 2 + 10000, 1)

# 计算 epsilon
epsilon = np.maximum(epsilon_finish, epsilon_start - delta * t)

anneal_right = 200000
anneal_left = 50000
delta_right = (epsilon_start - epsilon_finish) / anneal_right
delta_left = (epsilon_start - epsilon_finish) / anneal_left
epsilon_right = np.maximum(epsilon_finish, epsilon_start - delta_right * t)
epsilon_left = np.maximum(epsilon_finish, epsilon_start - delta_left * t)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(t, epsilon_right, label='epsilon_anneal_time=200000', color='red')
plt.plot(t, epsilon_left, label='epsilon_anneal_time=50000', color='green')
plt.plot(t, epsilon, label='default', color='blue')
plt.xlabel('时间步')
plt.ylabel(r'探索系数$\epsilon$')
plt.title(r'探索系数$\epsilon$衰减时间')

# 去除上方和右方的边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置网格线为虚线
plt.grid(True, linestyle='--')
plt.legend(prop={'family': 'Times New Roman'})
# plt.show()
plt.savefig('epsilon_anneal.pdf', dpi=300, bbox_inches='tight')
plt.close()


finish_more = 0.1
finish_less = 0.01
delta_right = (epsilon_start - finish_more) / epsilon_anneal_time
delta_left = (epsilon_start - finish_less) / epsilon_anneal_time
epsilon_right = np.maximum(finish_more, epsilon_start - delta_right * t)
epsilon_left = np.maximum(finish_less, epsilon_start - delta_left * t)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(t, epsilon_right, label='epsilon_finish=0.1', color='red')
plt.plot(t, epsilon_left, label='epsilon_finish=0.01', color='green')
plt.plot(t, epsilon, label='default', color='blue')
plt.xlabel('时间步')
plt.ylabel(r'探索系数$\epsilon$')
plt.title(r'探索系数$\epsilon$结束值')

# 去除上方和右方的边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置网格线为虚线
plt.grid(True, linestyle='--')
plt.legend(prop={'family': 'Times New Roman'})
# plt.show()
plt.savefig('epsilon_finish.pdf', dpi=300, bbox_inches='tight')
plt.close()


finish_more = epsilon_start - delta * 80000
finish_less = epsilon_start - delta * 104211
epsilon_right = np.maximum(finish_more, epsilon_start - delta * t)
epsilon_left = np.maximum(finish_less, epsilon_start - delta * t)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(t, epsilon_right, label='epsilon_left', color='red')
plt.plot(t, epsilon_left, label='epsilon_right', color='green')
plt.plot(t, epsilon, label='default', color='blue')
plt.xlabel('时间步')
plt.ylabel(r'探索系数$\epsilon$')
plt.title(r'保持探索衰减速率$\delta$不变')

# 去除上方和右方的边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置网格线为虚线
plt.grid(True, linestyle='--')

plt.legend(prop={'family': 'Times New Roman'})
# plt.show()
plt.savefig("epsilon_delta.pdf", dpi=300, bbox_inches='tight')
plt.close()