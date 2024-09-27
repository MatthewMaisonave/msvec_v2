import pandas as pd

def calculate_metrics(trueP, trueN, falseP, falseN):
    total = trueP + trueN + falseP + falseN
    accuracy = (trueP + trueN) / total
    precision = trueP / (trueP + falseP)  
    recall = trueP / (trueP + falseN)  
    f1 = 2 * ((precision * recall) / (precision + recall)) 

    print(f"TP: {trueP} TN: {trueN}\nFP: {falseP} FN: {falseN} \ntotal: {total}")
    print(f"accuracy = {format(accuracy*100, '.2f')}% \nprecision = {format(precision*100,'.2f')}%\nrecall = {format(recall*100,'.2f')}%\nf1 = {format(f1*100,'.2f')}%")
    results = {'accuracy': format(accuracy*100, '.2f'), 'precision':format(precision*100,'.2f'), 'recall': format(recall*100,'.2f'), 'f1':format(f1*100,'.2f')}
    return results

def evaluateStance_support(file_path):
    trueP = 0
    falseP = 0
    trueN = 0
    falseN = 0
    total = 0
    df = pd.read_csv(file_path)

    ground_truth = df['true/false'].astype(bool).tolist()
    labels = df['label'].tolist()
    ids = df['id'].astype(int).tolist()

    # Loop through both lists and calculate the counts
    for gt, label, id in zip(ground_truth, labels, ids):
        total += 1
        if gt and label == 'SUPPORT':
            trueP += 1
            print(f"{id} is a trueP, gt = {gt} and label = {label}")
        elif not gt and label == 'CONTRADICT': 
            trueN += 1
            print(f"{id} is a trueN, gt = {gt} and label = {label}")
        elif not gt  and (label == 'SUPPORT' or label == 'NEI'):
            falseP += 1
            print(f"{id} is a falseP, gt = {gt} and label = {label}")
        elif gt  and (label == 'CONTRADICT'or label == 'NEI'):
            falseN += 1
            print(f"{id} is a falseN, gt = {gt} and label = {label}")
        print(f"TP: {trueP} TN: {trueN}\nFP: {falseP} FN: {falseN} \ntotal: {total}")

    print()
    print('Support Class Stance:\n')
    results = calculate_metrics(trueP, trueN, falseP, falseN)
    return results

def evaluateStance_contradict(file_path):
    trueP = 0
    falseP = 0
    trueN = 0
    falseN = 0
    df = pd.read_csv(file_path)

    ground_truth = df['true/false'].astype(bool).tolist()
    labels = df['label'].tolist()

    # Loop through both lists and calculate the counts
    for gt, label in zip(ground_truth, labels):
        if gt and label == 'SUPPORT':
            trueN += 1
        elif not gt and label == 'CONTRADICT': 
            trueP += 1
        elif not gt  and (label == 'SUPPORT' or label == 'NEI'):
            falseN += 1
        elif gt  and (label == 'CONTRADICT'or label == 'NEI'):
            falseP += 1

    print()
    print('Contra Class Stance:\n')
    results = calculate_metrics(trueP, trueN, falseP, falseN)
    return results

def evaluateRationale_support(file_path, ground_truth_path):
    trueP = 0
    falseP = 0
    trueN = 0
    falseN = 0
    df = pd.read_csv(file_path)
    df_gt = pd.read_csv(ground_truth_path)
    # Exclude rows with IDs in the range [5601-5656]
    excluded_ids = df[df['id'].between(5601, 5656)]['id']
    df = df[~df['id'].between(5601, 5656)]
    df_gt = df_gt[~df_gt['id'].between(5601, 5656)]
    ground_truth = df_gt['support'].astype(float).tolist()
    rationale_values = df['GPT_Response_rationale'].astype(float).tolist()
    excluded_ids_present = any(df['id'].isin(excluded_ids)) or any(df_gt['id'].isin(excluded_ids))
    print("Excluded IDs present:", excluded_ids_present)
    # Loop through both lists and calculate the counts
    for gt, rationale in zip(ground_truth, rationale_values):
        if gt == 1.0 and rationale == 1.0:
            trueP += 1
        elif gt == 0.0 and rationale == 0.0:
            trueN += 1
        elif gt == 0.0 and rationale == 1.0:
            falseP += 1
        elif gt == 1.0 and rationale == 0.0:
            falseN += 1

    print()
    print('Support Class Rationale:\n')
    results = calculate_metrics(trueP, trueN, falseP, falseN)
    return results

def evaluateRationale_contradict(file_path, ground_truth_path):
    trueP = 0
    falseP = 0
    trueN = 0
    falseN = 0
    df = pd.read_csv(file_path)
    df_gt = pd.read_csv(ground_truth_path)
    # Exclude rows with IDs in the range [5601-5656]
    excluded_ids = df[df['id'].between(5601, 5656)]['id']
    df = df[~df['id'].between(5601, 5656)]
    df_gt = df_gt[~df_gt['id'].between(5601, 5656)]
    ground_truth = df_gt['support'].astype(float).tolist()
    rationale_values = df['GPT_Response_rationale'].astype(float).tolist()
    excluded_ids_present = any(df['id'].isin(excluded_ids)) or any(df_gt['id'].isin(excluded_ids))
    print("Excluded IDs present:", excluded_ids_present)
    # Loop through both lists and calculate the counts
    for gt, rationale in zip(ground_truth, rationale_values):
        if gt == 1.0 and rationale == 1.0:
            trueN += 1
        elif gt == 0.0 and rationale == 0.0:
            trueP += 1
        elif gt == 0.0 and rationale == 1.0:
            falseN += 1
        elif gt == 1.0 and rationale == 0.0:
            falseP += 1

    print()
    print('Contra Class Rationale:\n')
    results = calculate_metrics(trueP, trueN, falseP, falseN)
    return results




inFile_stance = 'parsed_stance.csv'
inFile_rationale = 'parsed_rationale_clean.csv'
ground_truth_path_rationale = 'gpt-3.5/data2/ground_truth_datasets/sentence_level_ground_truth-v1.csv' 

results_stance_support = evaluateStance_support(inFile_stance)
results_stance_contradict = evaluateStance_contradict(inFile_stance)
results_rationale = evaluateRationale_support(inFile_rationale, ground_truth_path_rationale)
results_rationale = evaluateRationale_contradict(inFile_rationale, ground_truth_path_rationale)

