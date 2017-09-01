import csv
from labMTsimple import storyLab
import re
from collections import defaultdict

class VerifyResult(object):

    def __init__(self):
        with open('meanscores.csv', 'r') as csvfile:
            self.wsudata = list(csv.reader(csvfile, delimiter=','))

        self.wsudict = defaultdict(float)
        for row in self.wsudata:
            self.wsudict[row[0].lower()] = float(row[1])

        with open('toverify.csv', 'r') as csvfile:
            self.inputfile = list(csv.reader(csvfile, delimiter=','))
        lang = 'english'
        self.labMTData, self.labMTSentiment, self.labMTwordList = storyLab.emotionFileReader(stopval=0.0, lang=lang, returnVector=True)


c = VerifyResult()
intersection_list = []
skiprow = True
for row in c.inputfile:
    if (skiprow):
        skiprow = False
        continue
    sum = 0
    count = 0

    tweet = re.sub(r'\W+', ' ', row[0]).lower()
    for word in tweet.split(' '):
        sum += c.wsudict[word]
        if (c.wsudict[word] > 0):
            count += 1

    if (count > 0):
        print(row[0].lower(), sum/count)


