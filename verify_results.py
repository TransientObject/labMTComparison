import csv
from labMTsimple import storyLab
import re
from collections import defaultdict
from nltk.stem.wordnet import WordNetLemmatizer


class SentimentDict(object):

    def lemmatized_word(self, word):
        if word[0] in ['#', '@']:
            return self.lmtzr.lemmatize(word[1:])
        return self.lmtzr.lemmatize(word)

    def __init__(self, optimize_labmt=True):
        with open('meanscores.csv', 'r') as csvfile:
            self.wsudata = list(csv.reader(csvfile, delimiter=','))

        self.wsudict = defaultdict(float)
        self.lmtzr = WordNetLemmatizer()
        lemmatized_list = defaultdict(lambda: [])

        for row in self.wsudata:
            self.wsudict[row[0].lower()] = float(row[1])
            lemmatized_list[self.lemmatized_word(row[0].lower())].append(row[0].lower())

        # recompute sentiment values by calculating average of all non-lemmatized word that belong to a lemmatized word
        for key, values in lemmatized_list.items():
            if len(values) > 1:
                lst = [self.wsudict[value] for value in values]
                average_sentiment_value = sum(lst)/len(lst)
                for value in values:
                    #print(",".join([value, str(self.wsudict[value]), str(average_sentiment_value), str(abs(self.wsudict[value]-average_sentiment_value))]))
                    self.wsudict[value] = average_sentiment_value

        #for key, value in lemmatized_list.items():
            #if len(value) > 1:
                #print(key, " : ", value )

        with open('toverify2.csv', 'r') as csvfile:
            self.inputfile = list(csv.reader(csvfile, delimiter=','))
        lang = 'english'
        self.labMTData, self.labMTSentiment, self.labMTwordList = storyLab.emotionFileReader(stopval=0.0, lang=lang, returnVector=True)

        if (optimize_labmt):
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

            #for key, value in lemmatized_list_labmt.items():
                #if len(value) > 1:
                    #print(key, " : ", value)

def GenerateTweetSentimentValues(combined = False):
    c = SentimentDict(False)
    intersection_list = []
    skiprow = True
    for row in c.inputfile:
        if (skiprow):
            skiprow = False
            continue
        sum = 0
        count = 0

        tweet = re.sub(r'\W+', ' ', row[0].strip()).lower().strip()
        for word in tweet.split(' '):
            try:
                if (c.wsudict[word] > 0):
                    sum += c.wsudict[word]
                    count += 1
                elif (c.wsudict[c.lemmatized_word(word)] > 0):
                    sum += c.wsudict[c.lemmatized_word(word)]
                    count+=1
                elif (combined and word in c.labMTData.keys() and float(c.labMTData[word][1]) > 0):
                    sum += float(c.labMTData[word][1])
                    count += 1
                elif (combined and c.lemmatized_word(word) in c.labMTData.keys() and float(c.labMTData[c.lemmatized_word(word)][1]) > 0):
                    sum += float(c.labMTData[c.lemmatized_word(word)][1])
                    count += 1
            except Exception as exc:
                print(exc)
                #print("tweet - ", tweet, "\n", "word - ", tweet.split(' '), "\n")

        if (count > 0):
            print(sum/count)


#GenerateTweetSentimentValues(True)

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

GenerateCSVOfLabMTWSUIntersectionWithDifferenceInSentimentValues()