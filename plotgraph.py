import csv
import matplotlib.pyplot as plt
import numpy as np

csvlist = []
with open('finalscores.csv', 'r') as csvfile:
    csvlist = list(csv.reader(csvfile, delimiter=','))

plot_input = []
for row in csvlist:
    plot_input.append(row[1:])

plot_input = sorted(plot_input, key=lambda x: x[-1], reverse=True)[:10]

print(plot_input)

#plt.hist([1,-2,3], 50, normed=1, facecolor='green', alpha=0.75)
fig = plt.figure()
ax1 = fig.add_axes([0, 0, .5, 2])
ax1.hist([1], 50, normed=1, facecolor='green', alpha=0.75)
#ax2 = fig.add_axes([5, 0, 5, 20])
plt.show()