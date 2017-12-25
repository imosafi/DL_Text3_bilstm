import numpy as np
import random
import datetime
import os
import matplotlib.pyplot as plt

with open('pos_a_evaluation.txt', 'r') as f:
    a = f.readline().strip().split()
    a = [float(x) for x in a]
with open('pos_b_evaluation.txt', 'r') as f:
    b = f.readline().strip().split()
    b = [float(x) for x in b]
with open('pos_c_evaluation.txt', 'r') as f:
    c = f.readline().strip().split()
    c = [float(x) for x in c]
with open('pos_d_evaluation.txt', 'r') as f:
    d = f.readline().strip().split()
    d = [float(x) for x in d]


plt.title('Net Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Sentences (x10000)')
plt.plot(a, color='tab:red')
plt.plot(b, color='tab:blue')
plt.plot(c, color='tab:green')
plt.plot(d, color='tab:brown')
plt.savefig('Accuracies.png', dpi=100)