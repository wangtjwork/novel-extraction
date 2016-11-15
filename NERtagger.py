import sys
import os
import re
os.environ["STANFORD_MODELS"] = "/Users/tianjianwang/Documents/Course_Study/2016_Fall/CSCE_689/FinalProject/stanford-ner-2015-12-09 "

from nltk.tag import StanfordNERTagger

st = StanfordNERTagger('/Users/tianjianwang/Documents/Course_Study/2016_Fall/CSCE_689/FinalProject/stanford-ner-2015-12-09/classifiers/english.all.3class.distsim.crf.ser.gz', '/Users/tianjianwang/Documents/Course_Study/2016_Fall/CSCE_689/FinalProject/stanford-ner-2015-12-09/stanford-ner.jar')
tagged = st.tag('Rami Eidd is studying at Stony Brook University in NY'.split())
res = []
for tags in tagged:
    if tags[1] == 'PERSON':
        res.append(tags[0])
print res