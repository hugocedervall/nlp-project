
# Author: Marco Kuhlman
class Dataset():

    ROOT = ('<root>', '<root>', 0)  # Pseudo-root

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, 'rt', encoding='utf-8') as lines:
            tmp = [Dataset.ROOT]
            for line in lines:
                if not line.startswith('#'):  # Skip lines with comments
                    line = line.rstrip()
                    if line:
                        columns = line.split('\t')
                        if columns[0].isdigit():  # Skip range tokens
                            tmp.append((columns[1], columns[3], int(columns[6])))
                    else:
                        yield tmp
                        tmp = [Dataset.ROOT]
