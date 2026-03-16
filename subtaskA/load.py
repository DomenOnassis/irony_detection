def parse_dataset(dataset):
    '''Loads the dataset .txt file with label-tweet on each line and parses the dataset.'''
    y = []
    corpus = []
    dataset_name = dataset.lower()
    with open(dataset, 'r', encoding='utf-8') as data_in:
        for line_num, line in enumerate(data_in):
            # Skip header line (always first line)
            if line_num == 0 and ("tweet index" in line.lower() or "label" in line.lower()):
                continue
            
            line = line.rstrip()	# remove trailing whitespace
            if not line.strip():  # skip empty lines
                continue
                
            if ("train" in dataset_name) or ("gold_test" in dataset_name):
                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        label = int(parts[1])
                        tweet = parts[2]
                        y.append(label)
                        corpus.append(tweet)
                    except (ValueError, IndexError):
                        continue
            else:
               parts = line.split("\t")
               if len(parts) >= 2:
                   tweet = parts[1]
                   corpus.append(tweet)
                   
    if ("train" in dataset_name) or ("gold_test" in dataset_name):
        return corpus, y
    else:
       return corpus