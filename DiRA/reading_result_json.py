import os
import json
import csv
import argparse

parser = argparse.ArgumentParser(description ='file names')
parser.add_argument('--results_path',required=True)
parser.add_argument('--results_filename',required=True)

args = parser.parse_args()

def is_json(filename):
    if filename.split(".")[-1] == 'json':
        return True
    else:
        return False

def reading_results(datapath):
    result_dicts = []
    count = 15
    for json_files in os.listdir(datapath):
        if not is_json(json_files):
            continue
        result = {}
        parsed_results_json = json.load(open(os.path.join(datapath, json_files)))
        result["Epoch"] = count
        result["Closed"] = parsed_results_json["CLOSED"]["score_percent"]
        result["Open"] = parsed_results_json["OPEN"]["score_percent"]
        result["Overall"] = parsed_results_json["ALL"]["score_percent"]
        result_dicts.append(result)
        count+=1
    
    print(result_dicts)
    return result_dicts

result_dicts = reading_results(args.results_path)

with open(os.path.join(args.results_path, args.results_filename), 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['Epoch','Closed','Open','Overall'])
    writer.writeheader()
    writer.writerows(result_dicts)
