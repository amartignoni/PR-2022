import csv
import re

with open('mol-save', 'rU') as txtfile:
    lines = txtfile.readlines()

matches = []

for line in lines:
    samples = re.findall("\.[a,i] [0-9]+", line)
    for sample in samples:
        sample_dot_removed = sample[1:]
        final_sample = sample_dot_removed.split(" ")
        print(final_sample)
        matches.append(final_sample)

print(matches, len(matches))

csv_entries = [[x[1], x[0]] for x in matches]

with open('mol.csv', 'w') as csvfile:

    writer = csv.writer(csvfile, delimiter=",")
    writer.writerows(csv_entries)