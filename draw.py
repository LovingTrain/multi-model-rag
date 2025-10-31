import re
import matplotlib.pyplot as plt
import numpy as np

def plot_fused(x,y1,y2,pic_name):

    plt.plot(x, y1, color='tab:blue', marker='o', label='retrieval with embedding')
    plt.plot(x, y2,  color='tab:orange',marker='o', label='retrieval')

    plt.xlabel('Batch Size (bsz)')
    plt.ylabel('Latency (ms)')

    plt.title(pic_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


x=np.array([100, 200 ,300  ,400  ,500  ,600  ,700  ,800  ,900  ,1000 ])

s= '''
0.03644
0.06120
0.07257
0.09524
0.13042
0.13534
0.15306
0.15484
0.16414
0.18625
'''
y1 = np.array([float(x) for x in s.strip().split('\n')])*1000

s2= '''
0.03644
0.06120
0.07257
0.09524
0.13042
0.13534
0.15306
0.15484
0.16414
0.18625
'''
y2 = np.array([float(x) for x in s2.strip().split('\n')])*1000

plot_fused(x,y1,y2+50,pic_name='prompt_length=128')
