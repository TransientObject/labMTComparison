import csv
from labMTsimple import storyLab
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def normalize(val):
    return 1.096*val + 5.27 #because vader scores range from -3.9 to 3.4

vaderAnalyzer = SentimentIntensityAnalyzer()

wsudata  = []
with open('meanscores.csv', 'r') as csvfile:
    wsudata = list(csv.reader(csvfile, delimiter=','))

wsudict = defaultdict(float)

for row in wsudata:
    wsudict[row[0].lower()] = float(row[1])

vaderDict = vaderAnalyzer.lexicon

commonWords = list(set(wsudict.keys()).intersection(vaderDict.keys()))
print("commonWords length", len(commonWords))

moverDict = defaultdict(lambda: [])

for word in commonWords:
    val = normalize(vaderDict[word]) - wsudict[word]
    moverDict[word] = [abs(val), val]

print("top 20 movers")
[print(x) for x in [(key, moverDict[key])for key in sorted(moverDict, key=moverDict.get, reverse=True)][0:20]]
