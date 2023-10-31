import os

def remove_files(path):
    not_remove_files = ["all_results.json", "generated_predictions.txt", "log", "info.json", "split.csv", "config.json"]
    for file_name in os.listdir(path):
        if file_name in not_remove_files:
            continue
        if file_name[-3:] == '.sh':
            continue
        if "checkpoint" in file_name:
            continue
        try:
            os.remove(os.path.join(path, file_name))
        except:
            print("Not remove {}".format(os.path.join(path, file_name)))

