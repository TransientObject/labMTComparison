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
#GenerateScatterPlot(1,7, "Human", "Vader With Regex")

def GenerateAccuracyPlot(x_index, y_index, x_neutral, y_neutral):
    with open('sentiment_scores.csv', 'r') as csvfile:
        sentiment_data = list(csv.reader(csvfile, delimiter=','))[1:]

    range = 0.5
    range_x = []
    accuracy_y = []
    range_x_ticks = []
    while(range <= 4.5):
        correct_classification = 0
        incorrect_classification = 0
        for row in sentiment_data:
            if(abs(float(row[x_index]) - x_neutral) > range):
                continue
            if (float(row[x_index]) > x_neutral and float(row[y_index]) > y_neutral) or (float(row[x_index]) < x_neutral and float(row[y_index]) < y_neutral):
                correct_classification += 1
            else:
                incorrect_classification += 1

        accuracy = correct_classification * 1.0 / (correct_classification + incorrect_classification)
        #print("range, accuracy", range, accuracy)
        range_x_ticks.append(str(4.5-range)+"-"+str(4.5+range))
        range_x.append(range)
        accuracy_y.append(accuracy)
        range += 0.5

    plt.plot(range_x, accuracy_y, 'go-')
    plt.ylabel("accuracy")
    plt.xlabel("human sentiment range")
    plt.xticks(range_x, range_x_ticks)
    plt.show()

GenerateAccuracyPlot(1,5,4.5,4.5)




