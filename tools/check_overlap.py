import json
labels = json.load(open("E:\Sign Language to Text\labels.json"))
print(len(labels), "labels total")
print(labels.get("3"))  # should be "Green"
