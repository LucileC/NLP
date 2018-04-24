import json

path = "msmarco_2wellformed/dev_v2.0_well_formed.json"

def loadDataset(path,limit=100000000000):
    print("...loading dataset")
    dataset = list()
    i = 0
    for line in open(path, 'r'):
        if i <3:
            data = json.loads(line)
            j = 0
            for d in data:
                if j<3:
                    print(d)
                    j += 1
            # print(data)
            # dataset.append(data)
            i += 1
        else:
            return dataset
    print("Loaded %d pieces of data"%len(dataset))
    return dataset

dataset = loadDataset(path,1000)
for i in range(5):
    print(dataset[i])