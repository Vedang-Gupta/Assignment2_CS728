import json
import numpy as np
import pickle
import random

with open("wiki_dict.json", "r", encoding="utf-8") as f:
    wiki_dict = json.load(f)
vals = list(wiki_dict.values())

c = 0
fever_data = []
def dataset_processing(file_path):
    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                claim = data['claim']
                evid_sets = data['evidence']
                evidences = []
                for evid_set in evid_sets:
                    evid_set = np.array(evid_set)[:, -2:]
                    for evid in evid_set:
                        if evid[0] != None:
                            evidences.append(wiki_dict[evid[0]].split("\n")[int(evid[1])])
                        else:
                            evidences.append(random.choice(vals).split("\n")[0])
                evidence = "[SEP]".join(evidences)
                claim_evid = "[CLS]" + claim + "[SEP]" + evidence
                # evid_set = data['evidence'][0]
                # evid_set = evid_set[0][-2:]
                # if evid_set[0] != None:
                #     evidence = wiki_dict[evid_set[0]].split("\n")[int(evid_set[1])]
                # else:
                #     evidence = random.choice(vals).split("\n")[0]
                if data['label'] == 'SUPPORTS':
                    label = 0
                elif data['label'] == 'REFUTES':
                    label = 1
                else:
                    label = 2
                fever_data.append((claim_evid, label))
    
            except Exception as e:
                if e == KeyError:
                    c += 1
                    print(e, c)
                    continue
            

dataset_processing("train.jsonl")

with open("fever_data_train_2.pkl", "wb") as f:
    pickle.dump(fever_data, f)
