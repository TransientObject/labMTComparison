import csv
from labMTsimple import storyLab
import re
from collections import defaultdict
from nltk.stem.wordnet import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentDict(object):

    def lemmatized_word(self, word):
        if word[0] in ['#', '@']:
            return word[1:]
        return word

    def __init__(self, optimize_dict=True):

        self.vaderAnalyzer = SentimentIntensityAnalyzer()

        with open('meanscores.csv', 'r') as csvfile:
            self.wsudata = list(csv.reader(csvfile, delimiter=','))

        self.wsudict = defaultdict(float)
        self.lmtzr = WordNetLemmatizer()
        lemmatized_list = defaultdict(lambda: [])

        for row in self.wsudata:
            self.wsudict[row[0].lower()] = float(row[1])
            if (optimize_dict):
                lemmatized_list[self.lemmatized_word(row[0].lower())].append(row[0].lower())

        self.labMTData, self.labMTSentiment, self.labMTwordList = storyLab.emotionFileReader(stopval=0.0, lang='english', returnVector=True)

        with open('toverify2.csv', 'r') as csvfile:
            self.inputfile = list(csv.reader(csvfile, delimiter=','))

        # for key, value in lemmatized_list.items():
        # if len(value) > 1:
        # print(key, " : ", value )
        # if len(value) == 1 and value[0] != key:
        # print(key, " : ", value)

        # recompute sentiment values by calculating average of all non-lemmatized word that belong to a lemmatized word
        if (optimize_dict):
            for key, values in lemmatized_list.items():
                if len(values) > 1:
                    lst = [self.wsudict[value] for value in values]
                    average_sentiment_value = sum(lst)/len(lst)
                    for value in values:
                        #print(",".join([value, str(self.wsudict[value]), str(average_sentiment_value), str(abs(self.wsudict[value]-average_sentiment_value))]))
                        self.wsudict[value] = average_sentiment_value

            lemmatized_list_labmt = defaultdict(lambda: [])

            for word in self.labMTwordList:
                lemmatized_list_labmt[self.lemmatized_word(word.lower())].append(word.lower())

            # recompute sentiment values by calculating average of all non-lemmatized word that belong to a lemmatized word
            for key, values in lemmatized_list_labmt.items():
                if len(values) > 1:
                    lst = [float(self.labMTData[value][1]) for value in values]
                    average_sentiment_value = sum(lst) / len(lst)
                    for value in values:
                        # print(",".join([value, str(self.wsudict[value]), str(average_sentiment_value), str(abs(self.wsudict[value]-average_sentiment_value))]))
                        self.labMTData[value][1] = average_sentiment_value

    def GetVaderSentimentScore(self, tweet):
        #words = [x.strip() for x in re.findall(r"[\w\@\#\'\&\]\*\-\/\[\=\;\!\?]+", tweet, flags=re.UNICODE)]
        #vaderScores = self.vaderAnalyzer.polarity_scores(' '.join(list(filter(None, words))))
        vaderScores = self.vaderAnalyzer.polarity_scores(tweet)
        return 4.0 * (vaderScores['compound']) + 5.0

    def GetWSUSentimentScore(self, tweet):
        sum = 0
        count = 0
        words = [x.lower() for x in re.findall(r"[\w\@\#\'\&\]\*\-\/\[\=\;]+", tweet, flags=re.UNICODE)]
        for word in words:
            if (len(word) == 0):
                continue
            if (self.wsudict[word] > 0):
                sum += self.wsudict[word]
                count += 1

        if (count > 0):
            return(sum/count)
        return 0

    def GetLabMTSentimentScore(self, tweet):
        sum = 0
        count = 0
        words = [x.lower() for x in re.findall(r"[\w\@\#\'\&\]\*\-\/\[\=\;]+", tweet, flags=re.UNICODE)]
        for word in words:
            if (len(word) == 0):
                continue
            if (word in self.labMTData.keys() and float(self.labMTData[word][1]) > 0):
                sum += float(self.labMTData[word][1])
                count += 1

        if (count > 0):
            return(sum/count)
        return 0

    def GetCombinedSentimentScore(self, tweet):
        sum = 0
        count = 0
        words = [x.lower() for x in re.findall(r"[\w\@\#\'\&\]\*\-\/\[\=\;]+", tweet, flags=re.UNICODE)]
        for word in words:
            if (len(word) == 0):
                continue
            if (self.wsudict[word] > 0):
                sum += self.wsudict[word]
                count += 1
            elif (word in self.labMTData.keys() and float(self.labMTData[word][1]) > 0):
                sum += float(self.labMTData[word][1])
                count += 1

        if (count > 0):
            return(sum/count)
        return 0

    def GetTravelMTSentimentScore(self, tweet):
        sum = 0
        count = 0
        words = [x.lower() for x in re.findall(r"[\w\@\#\'\&\]\*\-\/\[\=\;]+", tweet, flags=re.UNICODE)]
        for word in words:
            if (len(word) == 0):
                continue
            word = self.lemmatized_word(word)
            if (self.wsudict[word] > 0):
                sum += self.wsudict[word]
                count += 1
            elif (word in self.labMTData.keys() and float(self.labMTData[word][1]) > 0):
                sum += float(self.labMTData[word][1])
                count += 1

        if (count > 0):
            return (sum / count)
        return 0

def GenerateTweetSentimentValues(mode = 'wsu', optimize_dict = False):
    c = SentimentDict(optimize_dict)
    skiprow = True
    for row in c.inputfile:
        if (skiprow):
            skiprow = False
            continue

        tweet = row[0]
        if (mode == 'wsu'):
            print(c.GetWSUSentimentScore(tweet))
        elif (mode == 'labmt'):
            print(c.GetLabMTSentimentScore(tweet))
        elif (mode == 'combined'):
            print(c.GetCombinedSentimentScore(tweet))
        elif (mode == 'travelmt'):
            print(c.GetTravelMTSentimentScore(tweet))
        elif (mode == 'vader'):
            print(c.GetVaderSentimentScore(tweet))

#GenerateTweetSentimentValues('vader', False)

def GenerateCSVOfLabMTWSUIntersection():
    dic = SentimentDict()
    for key in dic.wsudict.keys():
        if key in dic.labMTwordList:
            print(key+","+"Yes")
        else:
            print(key+","+"No")

#GenerateCSVOfLabMTWSUIntersection()

def GenerateCSVOfLabMTWSUIntersectionWithDifferenceInSentimentValues():
    dic = SentimentDict(False)
    for key in dic.wsudict.keys():
        if key in dic.labMTwordList:
            print(key+","+str(dic.labMTData[key][1])+","+str(dic.wsudict[key]))

#GenerateCSVOfLabMTWSUIntersectionWithDifferenceInSentimentValues()