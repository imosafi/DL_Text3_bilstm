import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# with open('pos_a_evaluation.txt', 'r') as f:
#     a = f.readline().strip().split()
#     a = [float(x) for x in a]
# with open('pos_b_evaluation.txt', 'r') as f:
#     b = f.readline().strip().split()
#     b = [float(x) for x in b]
# with open('pos_c_evaluation.txt', 'r') as f:
#     c = f.readline().strip().split()
#     c = [float(x) for x in c]
# with open('pos_d_evaluation.txt', 'r') as f:
#     d = f.readline().strip().split()
#     d = [float(x) for x in d]

with open('ner_a_evaluation.txt', 'r') as f:
    a = f.readline().strip().split()
    a = [float(x) for x in a]
with open('ner_b_evaluation.txt', 'r') as f:
    b = f.readline().strip().split()
    b = [float(x) for x in b]
with open('ner_c_evaluation.txt', 'r') as f:
    c = f.readline().strip().split()
    c = [float(x) for x in c]
with open('ner_d_evaluation.txt', 'r') as f:
    d = f.readline().strip().split()
    d = [float(x) for x in d]


plt.title('Net Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Sentences')
red_patch = mpatches.Patch(color='red', label='a')
blue_patch = mpatches.Patch(color='blue', label='b')
green_patch = mpatches.Patch(color='green', label='c')
brown_patch = mpatches.Patch(color='brown', label='d')
plt.legend(handles=[red_patch, blue_patch, green_patch, brown_patch], loc=4)
plt.plot([5 * i for i in range(0, len(a))], a, color='tab:red')
plt.plot([5 * i for i in range(0, len(b))], b, color='tab:blue')
plt.plot([5 * i for i in range(0, len(c))], c, color='tab:green')
plt.plot([5 * i for i in range(0, len(d))], d, color='tab:brown')
plt.savefig('Accuracies.png', dpi=100)