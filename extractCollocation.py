#get PMI index
class PMI3:
    def __init__(self, **kargs):
        self.dictCount = {}
        self.dictTriCount = {}
        self.nTotal = 0

    def train(self, sentenceIter, weight = 1):
        for sent in sentenceIter:
            self.nTotal += len(sent)
            for word in sent:
                self.dictCount[word] = self.dictCount.get(word, 0) + weight
            for a, b, c in zip(sent[:-2], sent[1:-1], sent[2:]):
                self.dictTriCount[a, b, c] = self.dictTriCount.get((a, b, c), 0) + weight

    def getCoOccurrence(self, a, b, c):
        return self.dictTriCount.get((a, b, c), 0)

    def getPMI(self, a, b, c):
        import math
        co = self.getCoOccurrence(a, b, c)
        if not co: return None
        return math.log(float(co) * self.nTotal * self.nTotal / self.dictCount[a] / self.dictCount[b] / self.dictCount[c])

    def getNPMI(self, a, b, c):
        import math
        abc = self.getPMI(a, b, c)
        if abc == None: return -1
        return abc / (2 * math.log(self.nTotal / self.getCoOccurrence(a, b, c)))

    def getPMIDict(self, minNum = 5):
        ret = {}
        for a, b, c in self.dictTriCount:
            if self.dictTriCount[a, b, c] < minNum: continue
            ret[a, b, c] = self.getPMI(a, b, c)
        return ret

    def getNPMIDict(self, minNum = 5):
        ret = {}
        for a, b, c in self.dictTriCount:
            if self.dictTriCount[a, b, c] < minNum: continue
            ret[a, b, c] = self.getNPMI(a, b, c)
        return ret

class SentenceReader:
    def __init__(self, filepath):
        self.filepath = filepath
  
    def __iter__(self):
        for line in open(self.filepath, encoding='utf-8'):
            yield list(s.split('\t'))


# Read from test.txt
pc = PMI3()
pc.train(SentenceReader('test.txt'))
# The normalized informations are calculated and printed in descending order for collocation appearing five or more times.
res = pc.getNPMIDict()
for a, b, c in sorted(res, key=res.get, reverse=True):
    print(a, b, c, res[a, b, c])
