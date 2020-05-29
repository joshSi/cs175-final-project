import os, re

extension = 'csv'
input_files = [f for f in os.listdir(".") if re.search(r"^\d{10,}\.csv$", f)]

header_saved = False
with open('combined.csv','w') as fout:
    for filename in input_files:
        print('Adding', filename)
        with open(filename) as fin:
            header = next(fin)
            if not header_saved:
                fout.write(header)
                header_saved = True
            for line in fin:
                fout.write(line)