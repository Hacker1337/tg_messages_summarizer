# %%
import pandas as pd
import os
import csv
from datasets import load_dataset
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict
from deep_translator import GoogleTranslator


dataset = load_dataset("knkarthick/dialogsum")


file_prefix = "output"

dry_run = False

columns = ['id', 'dialogue', 'summary', 'topic']

for key in list(dataset.keys()):
    subset = dataset[key]

    file_name = file_prefix + "_" + key + ".csv"

    resuming = os.path.exists(file_name)

    csvfile = open(file_name, "a", newline="")
    csvwriter = csv.writer(csvfile)

    if resuming:
        n_processed = len(pd.read_csv(file_name))
    else:
        n_processed = 0
        csvwriter.writerow(columns)

    for j in tqdm(range(n_processed, len(subset)), desc=key):

        entry = subset[j]
        for column in ['dialogue', 'summary', 'topic']:
            to_translate = entry[column]
            done = False
            n_tries = 0
            max_tries = 10
            while not done:
                try:
                    translated = GoogleTranslator(source='en', target='ru').translate(to_translate)
                    done = True
                except Exception as e:
                    print("got exception:\n", e)
                    n_tries += 1
                    if n_tries > max_tries:
                        break
            if n_tries > max_tries:
                print("Warning: skipped entry", entry["id"], "because it is too big or something else")
                continue
            entry[column] = translated
        csvwriter.writerow([entry[x] for x in columns])

        if dry_run:
            break

    csvfile.close()

# %%
