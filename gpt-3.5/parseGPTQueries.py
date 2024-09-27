import pandas as pd
import re

# Function to parse GPT_Response_1 column and add columns to data frames
def parse_gpt_response_rationale(row):
    response = str(row['GPT_Response_1'])    
    pattern_t1 = re.compile(r'T1: (\w+) (\d+) Confidence: ([\d.]+)')
    pattern_t2 = re.compile(r'T2: ([\d, ]+) Confidence: ([\d.]+)')
    
    # Search for T1
    match_t1 = pattern_t1.search(response)
    if match_t1:
        row['label'] = match_t1.group(1)
        row['relevance'] = int(match_t1.group(2))
        row['confidence'] = float(match_t1.group(3))

    # Search for T2
    match_t2 = pattern_t2.search(response)
    if match_t2:
        pass
        selected_sentences = [int(num) for num in match_t2.group(1).split(',')]
        row['selected_sentence_list'] = str(selected_sentences).replace("'", "")
        row['confidence_t2'] = float(match_t2.group(2))
    
    return row

def rationale_scorer(df):
    claimIDs = [] 
    GPT_Response = df['GPT_Response_1'].tolist()
    i = 0
    current = 0
    #print(GPT_Response)
    for index, row in df.iterrows():
        
        claimIDs.append(row['id']) # For referencing next claim id during iteration

    for index, row in df.iterrows():
        
        currentID = row['id']
        
        if(index <= (len(claimIDs) - 2)):
            nextID = claimIDs[index + 1]
            
        if(index == (len(claimIDs) - 1)):
            nextID = 0
        
        if str(i) in GPT_Response[current]:
            df.at[index, 'GPT_Response_rationale'] = 1
            
        else:
            df.at[index, 'GPT_Response_rationale'] = 0
            
        i += 1
        
        if(nextID != currentID):
            i = 0
            current += 1
            
    df.to_csv(outFile,index=False)

def temp(df2,df3):
    df2['GPT_Response_1'] = df2['GPT_Response_1'].replace('\n', '\'n', regex=True)
    df2['GPT_Response_1_Part2'] = df2['GPT_Response_1'].str.split('\'n').str[1]
    df3['GPT_Response_1_Part1'] = df2['GPT_Response_1'].str.split('\'n').str[0]
    df3['GPT_Response_1'] = df2['GPT_Response_1']
    df3 = df3.apply(parse_gpt_response_rationale, axis=1)
    desired_cols = ['id', 'title', 'claim', 'Type', 'url', 'domain', 'news_date', 'true/false', 'published_paper_urls', 'published_paper_title', 'published_paper_authors', 'published_paper_abstract', 'published_paper_venue', 'published_paper_year', 'Answers', 'Answers.1', 'Answers.2', 'Prompt', 'GPT_Response_1_Part1', 'GPT_Response_1', 'label', 'relevance', 'confidence', 'selected_sentence_list', 'confidence_t2']
    df3 = df3[desired_cols]
    df3.to_csv(outFile3, index=False)
    non_empty_responses = df2['GPT_Response_1_Part2'].dropna().tolist()
    df2 = df2.drop(columns=['GPT_Response_1', 'GPT_Response_1_Part2'])
    df2['GPT_Response_1'] = ''
    first_occurrences = df2.drop_duplicates(subset=['id'], keep='first').index
    for idx, response in zip(first_occurrences, non_empty_responses):
        df2.at[idx, 'GPT_Response_1'] = response

    df2.to_csv(outFile2, index=False)



outFile = 'parsed_rationale.csv'
outFile2 = 'parsed_rationale_clean.csv'
outFile3 = 'parsed_stance.csv'
inFile = 'gpt-3.5/data2/indexed_rationale_responses.csv'
inFile2 = 'gpt-3.5/data2/ground_truth_datasets/claim_level_ground_truth-v1.csv'

df = pd.read_csv(inFile)
rationale_scorer(df)
df2 = pd.read_csv(outFile,usecols=['id', 'title', 'claim', 'url', 'domain', 'news_date', 'true/false','published_paper_urls', 'published_paper_title', 'published_paper_authors','published_paper_abstract', 'published_paper_venue', 'published_paper_year','GPT_Response_1','GPT_Response_rationale'])
df3 = pd.read_csv(inFile2)
temp(df2,df3)



        

        