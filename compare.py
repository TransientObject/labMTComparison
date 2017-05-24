import csv
from labMTsimple import storyLab
import re

class Compare(object):

    def __init__(self):
        with open('meanscores.csv', 'r') as csvfile:
            self.csvlist = list(csv.reader(csvfile, delimiter=','))
        lang = 'english'
        self.labMTData, self.labMTSentiment, self.labMTwordList = storyLab.emotionFileReader(stopval=0.0, lang=lang, returnVector=True)


c = Compare()
intersection_list = []
for row in c.csvlist:
    word = re.sub(r'\W+', '', row[0]).lower() # there are some words like #Hungary. To compare better, stripping all non alpha numeric chars and converting it into lower
    if word in c.labMTwordList:
        intersection_list.append([word, row[1], c.labMTData[word][1]]) #appending [word, csv_score, labMTScore]

print(intersection_list)





