import os
os.environ["STANFORD_MODELS"] = "/Users/tianjianwang/Documents/Course_Study/2016_Fall/CSCE_689/FinalProject/stanford-ner-2015-12-09 "

class FileProcess:
    
    def __init__(self, fileName):
        self.fileName = fileName
        
    def getMainCharacter(self,amount):

        from nltk.tag import StanfordNERTagger
        
        st = StanfordNERTagger('/Users/tianjianwang/Documents/Course_Study/2016_Fall/CSCE_689/FinalProject/stanford-ner-2015-12-09/classifiers/english.all.3class.distsim.crf.ser.gz', '/Users/tianjianwang/Documents/Course_Study/2016_Fall/CSCE_689/FinalProject/stanford-ner-2015-12-09/stanford-ner.jar')
        
        mem = {}
        i = 0.0
        
        totalLine = self.mapcount()
        print "Total Line: {}".format(totalLine)
        with open(self.fileName, 'r') as f:
            for line in f:
                i += 1
                print str(i/totalLine * 100) + "%"
                if not line == '\n' and not line == '\r\n': #check if the line is an empty one
                    wordbuff = st.tag(line.split())
                    for word, tag in wordbuff:
                        if tag == "PERSON":
                            mem[word] = mem.get(word,0) + 1
            
        charNames = [(mem[word], word) for word in mem]
        
        charNames.sort()
        
        for time, name in charNames[:-1 * amount - 1:-1]:
            print "Name {} appeared {} times".format(name, time)
        
    def filterFile(self, name, bufName):
        name = name.split(' ')
        firstname = name[0]
        lastname = name[-1]
        buf = open(bufName, 'w')
        with open(self.fileName, 'r') as f:
            for line in f:
                if firstname in line.split():
                    buf.write(line)
        return
    
    def mapcount(self):
        import mmap
        f = open(self.fileName, "r+")
        buf = mmap.mmap(f.fileno(), 0)
        lines = 0
        readline = buf.readline
        while readline():
            lines += 1
        return lines

            