import json

jsondata = {
    "word2vecfile": "/home/anustup/Desktop/Mediadistillery/Chapter_detection/data/GoogleNews-vectors-negative300.bin",
    "choidataset": "/home/anustup/Desktop/Mediadistillery/Chapter_detection/text-segmentation/data/choi",
    "wikidataset": "/home/anustup/Desktop/Mediadistillery/Chapter_detection/data/wiki_727",
}

with open('config.json', 'w') as f:
    json.dump(jsondata, f)
