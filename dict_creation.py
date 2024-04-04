import json
import os 

wiki_dump_dir = "wiki-pages/wiki-pages"
# file_path = "wiki-pages/wiki-pages/wiki-001.jsonl"

wiki_dict = {}
for file_name in os.listdir(wiki_dump_dir):
    with open(os.path.join(wiki_dump_dir, file_name), "r", encoding="utf-8") as f:
        for line in f:
            json_obj = json.loads(line.strip())
            wiki_dict[json_obj['id']] = json_obj['lines']

with open("wiki_dict.json", "w",encoding="utf-8") as f:
    json.dump(wiki_dict, f)


        