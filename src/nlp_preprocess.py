import re
import itertools

def preprocess(sentence):
    sentence=str(sentence)
    filtered_words = re.sub('\W+',' ', sentence)
    
    return filtered_words



 
def getMaxOccurrence(stringsList, key):
    count = 0
    for word in stringsList:
        if key in word:
            count += 1
    return count

def getSubSequences(STR):
    combs = []
    result = []
    for l in range(1, len(STR)+1):
        combs.append(list(itertools.combinations(STR, l)))

    for c in combs:
        for t in c:
            result.append(''.join(t))
    return result

def getCommonSequences(S):
    mainList = []
    for word in S:
        temp = getSubSequences(word)
        mainList.extend(temp)
    
    mainList = list(set(mainList))
    mainList = reversed(sorted(mainList, key=len))
    mainList = list(filter(None, mainList))

    finalData = dict()

    for alpha in mainList:
        val = getMaxOccurrence(S, alpha)
        if val > 0:
            finalData[alpha] = val

    finalData = {k: v for k, v in sorted(finalData.items(), key=lambda item: item[1], reverse=True)}

    return finalData