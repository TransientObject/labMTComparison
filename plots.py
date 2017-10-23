import numpy as np
import matplotlib.pyplot as plt
import csv

def GenerateHistogram():
    with open('sentiment_scores.csv', 'r') as csvfile:
        sentiment_data = list(csv.reader(csvfile, delimiter=','))

    plot_input_list = []
    skipRow = True
    for row in sentiment_data:
        if skipRow:
            skipRow = False
            continue
        plot_input_list.append([float(x) for x in row[4:]])

    colors = ['blue', 'yellow']
    labels = ['labmt', 'travelmt']
    plt.hist(np.array(plot_input_list), 9, normed=1, histtype='bar', color=colors, label=labels)
    plt.legend(prop={'size': 9})
    # plt.title('Histogram with Sentiment values in LabMT and TravelMT \n', fontsize = 12, fontweight = 'bold', )
    plt.show()

#GenerateHistogram()

def GenerateScatterPlot(x_index, y_index, x_label, y_label):
    with open('sentiment_scores.csv', 'r') as csvfile:
        sentiment_data = list(csv.reader(csvfile, delimiter=','))

    plot_x = []
    plot_y = []
    skipRow = True
    for row in sentiment_data:
        if skipRow:
            skipRow = False
            continue
        plot_x.append(float(row[x_index]))
        plot_y.append(float(row[y_index]))

    plt.scatter(plot_x, plot_y, alpha=0.5)
    plt.legend(prop={'size': 9})
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.plot([int(min(plot_x))-1,int(max(plot_x))+1], [int(min(plot_x))-1,int(max(plot_x))+1])
    plt.plot([int(min(plot_x)) - 1, int(max(plot_x))], [int(min(plot_x)), int(max(plot_x)) + 1], '--', color = "orange")
    plt.plot([int(min(plot_x)), int(max(plot_x)) + 1], [int(min(plot_x)) - 1, int(max(plot_x))], '--', color = "orange")
    plt.title("Sentiment score across all tweets : "+ x_label +" and "+ y_label, fontweight = 'bold', fontsize = 14 )
    plt.show()

#GenerateScatterPlot(1,5, "Human", "TravelMT")
#GenerateScatterPlot(2,5, "LabMT", "TravelMT")
#GenerateScatterPlot(1,2, "Human", "LabMT")

#GenerateScatterPlot(1,6, "Human", "Vader")
GenerateScatterPlot(1,7, "Human", "Vader With Regex")