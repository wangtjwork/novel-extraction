import json, os

with open('data/signalmedia-1m.jsonl', 'r') as f:
    article = open('data/article/article.txt', 'w')
    title = open('data/title/title.txt', 'w')
    
    for _ in range(100):
        line = f.readline()
        jfile= json.loads(line)
    
        article.write(jfile['content'].encode("utf-8") + '\n')
        title.write(jfile['title'].encode("utf-8") + '\n')