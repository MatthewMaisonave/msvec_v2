import time, os, openai, re
import pandas as pd

#api key stored as environment variable for security
openai.api_key = os.getenv('OPENAI_API_KEY')
file_path   = 'gpt-3.5/data2/ground_truth_datasets/domain/'
gt_path = 'gpt-3.5/data2/ground_truth_datasets/sentence_level_ground_truth-v1_domain.csv'
export_path = 'gpt-3.5/data2/rationale/zero_shot/'

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0, # This is the degree of randomness of the model's output
    )
    return response.choices[0].message.content


def rationale_scorer(df, domain):
    claimIDs = [] 
    GPT_Response = df['GPT_Response_1'].tolist()
    i = 0
    current = 0
    #print(GPT_Response)
    for index, row in df.iterrows():
        
        claimIDs.append(row['id']) # For referencing next claim id during iteration

    nextID = None
    for index, row in df.iterrows():
        
        currentID = row['id']
        
        if(index <= (len(claimIDs) - 2)):
            nextID = claimIDs[index + 1]
            
        if(index == (len(claimIDs) - 1)):
            nextID = 0
        
        if str(i) in str(GPT_Response[current]):
            df.at[index, 'GPT_Response_rationale'] = 1
            
        else:
            df.at[index, 'GPT_Response_rationale'] = 0
            
        i += 1
        
        if(nextID != currentID):
            i = 0
            current += 1
            
    #df = df.apply(parse_gpt_response_rationale, axis=1)
    df.to_csv(export_path + 'rationale_' + domain,index=False)
    

    
    
    
    
    
    
def evaluateRationale_support(file_path, ground_truth_path):
    trueP = 0
    falseP = 0
    trueN = 0
    falseN = 0
    df = pd.read_csv(file_path)
    df_gt = pd.read_csv(ground_truth_path, usecols=['id','claim','domain','support'])
    #exclude each occurence of the first 4 unique IDs in the gt and any IDs that don't match the domain
    unique_ids = df['id'].unique()[:4]
    df_gt = df_gt[~df_gt['id'].isin(unique_ids)]
    excluded_ids = df_gt['id'].unique()[:4]
    ground_truth = df_gt['support'].astype(float).tolist()
    rationale_values = df['GPT_Response_rationale'].astype(float).tolist()
    excluded_ids_present = any(df['id'].isin(excluded_ids)) or any(df_gt['id'].isin(excluded_ids))
    print("Excluded IDs present:", excluded_ids_present)
    #print both lists of values and their IDs side by side
    print("Ground Truth:", ground_truth)
    print("Rationale Values:", rationale_values)
    print("IDs:", df['id'].tolist())
    print("GT IDs:", df_gt['id'].tolist())
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

def calculate_metrics(trueP, trueN, falseP, falseN):
    total = trueP + trueN + falseP + falseN
    accuracy = (trueP + trueN) / total
    precision = trueP / (trueP + falseP)  
    recall = trueP / (trueP + falseN)  
    f1 = 2 * ((precision * recall) / (precision + recall)) 

    print(f"TP: {trueP} TN: {trueN}\nFP: {falseP} FN: {falseN} \ntotal: {total}")
    print(f"accuracy = {format(accuracy*100, '.2f')}% \nprecision = {format(precision*100,'.2f')}%\nrecall = {format(recall*100,'.2f')}%\nf1 = {format(f1*100,'.2f')}%")
    results = {'accuracy': format(accuracy*100, '.2f'), 'precision':format(precision*100,'.2f'), 'recall': format(recall*100,'.2f'), 'f1':format(f1*100,'.2f')}
    with open(export_path + "_results.txt", "w") as output:
        output.write(f"TP: {trueP} TN: {trueN}\nFP: {falseP} FN: {falseN} \ntotal: {total}") 
        output.write(f"accuracy = {format(accuracy*100, '.2f')}% \nprecision = {format(precision*100,'.2f')}%\nrecall = {format(recall*100,'.2f')}%\nf1 = {format(f1*100,'.2f')}%")
    return results

def shot_query(domain,prompts):
    promptsDF = pd.DataFrame(prompts, columns=['prompts'])
    responses = []
    requests = 0
    col_name = 'GPT_Response_1'
    for index,prompt in enumerate(prompts):
        if requests>0 and requests%70 == 0:
            print("Sleeping for 70 seconds...") 
            time.sleep(70)
            print("Continuing...currently on request " + str(requests))

        requests += 1

        responses.append(get_completion(prompt))
        promptsDF.at[index, col_name] = responses[index] # Name of column in output file 
    
    # Export to CSV(creates CSV with additional column(s))
    promptsDF.to_csv(export_path + domain)
    print("Done with export " + domain + "!\n")
    #return promptsDF
    
    
def attach_supporting_sentences(unindexed_df, indexed_df):
    # Initialize a list to store the result for each row in unindexed_df
    selected_sentence_lists = []

    # Loop through each row in the unindexed DataFrame
    for _, u_row in unindexed_df.iterrows():
        current_id = u_row['id']
        sentences_indices = []

        # Initialize a counter to track the local sentence index within the same ID
        local_index = 0
        previous_id = None

        # Loop through each row in the indexed DataFrame to find matching IDs
        for _, i_row in indexed_df.iterrows():
            if i_row['id'] == current_id:
                if previous_id != current_id:
                    local_index = 0  # Reset local index for a new ID group
                if i_row['support'] == 1:
                    sentences_indices.append(local_index)
                local_index += 1
            previous_id = i_row['id']

        # Store the list of indices for this particular ID
        selected_sentence_lists.append(sentences_indices)

    # Assign the collected indices back to the unindexed dataframe
    unindexed_df['selected_sentence_list'] = selected_sentence_lists
    return unindexed_df

# Function to parse GPT_Response_1 column and add columns to data frames
def parse_gpt_response_rationale(row):
    #response can be a list of numbers with commas and spaces or a list of numbers with commas and spaces and brackets or a list of numbers with commas and spaces and brackets and quotes 
    response = row['GPT_Response_1']
    pattern = re.compile(r'([\d, ]+)')
    match = pattern.search(response)
    if match:
        selected_sentence_list = [int(num) for num in match.group().split(',')]
        row['selected_sentence_response'] = str(selected_sentence_list).replace("'", "") #replace single quotes with nothing
    
    return row



def start_up():
    inFile = 'gpt-3.5/data2/ground_truth_datasets/sentence_level_ground_truth-v1_domain.csv'
    inFile_unindexed = 'gpt-3.5/data2/ground_truth_datasets/claim_level_ground_truth-v1_for_rationale.csv'
    inDir_domain = 'gpt-3.5/data2/ground_truth_datasets/domain/'    
    outDir_domain = 'gpt-3.5/data2/ground_truth_datasets/domain_rationale/'
    
    def generate_gt_rationales():
        df = pd.read_csv(inFile)
        df2 = pd.read_csv(inFile_unindexed, usecols=['id', 'title', 'claim', 'Type', 'url', 'domain', 'true/false', 'published_paper_urls', 'published_paper_title', 'published_paper_abstract'])
        
        i=0
        claimIDs = [] 
        abstracts = []
        abstractsNoBrackets = []
        prompts = []
        
        for index, row in df.iterrows():
            # Used to compare next id in index during iteration
            claimIDs.append(row['id'])
            
        for index, row in df.iterrows():
            currentID = row['id']
            if(index <= (len(claimIDs) - 2)):
                nextID = claimIDs[index + 1]

            if(index == (len(claimIDs) - 1)):
                nextID = 0

            abstracts.append(str(i)+". "+row['published_paper_abstract'] + "\n")
            abstractsNoBrackets = ''.join(abstracts)
            i += 1
            if (nextID != currentID):
                prompts.append(f"Claim: {str(row['claim'])}\n" + f"Abstract:\n{abstractsNoBrackets}")
                abstracts = []
                abstractsNoBrackets = []
                i = 0

        promptsDF = pd.DataFrame(prompts, columns=['prompts'])
        promptsDF2 = pd.DataFrame(prompts, columns=['prompts'])
        df = pd.concat([df, promptsDF], axis=1)
        df2 = pd.concat([df2, promptsDF2], axis=1)
        
        df2 = attach_supporting_sentences(df2, df)
        df2.to_csv(outDir_domain + 'unindexed_rationale_gt.csv', index=False)
        df.to_csv(outDir_domain + 'indexed_rationale_gt.csv', index=False)
              
    def generate_gt_domains():
        os.makedirs(outDir_domain, exist_ok=True)

        # Load the CSV file into a DataFrame
        df = pd.read_csv(outDir_domain + 'unindexed_rationale_gt.csv')
        for domain, group in df.groupby('domain'):
            # Define the filename based on the domain
            filename = f"{domain}_domain.csv"
            # Save each group to a separate CSV file
            group.to_csv(outDir_domain + filename, index=False)
        
        
    generate_gt_rationales()
    generate_gt_domains()
    
    
  
    
    
    
def prompt_generation():
    file_path = 'gpt-3.5/data2/ground_truth_datasets/domain_rationale/domain/'
    num_runs = 1
    
    def zero_shot():
        export_path = 'gpt-3.5/data2/rationale/zero_shot/'
        df_gt = pd.read_csv('gpt-3.5/data2/ground_truth_datasets/domain_rationale/unindexed_rationale_gt.csv')
        for file in os.listdir(file_path):
            if file.endswith(".csv") and (file in ['health_domain.csv','humans_domain.csv']):
                domain = file
                df = pd.read_csv(file_path + domain)
                df_training = df.head(4)
                df_testing = df.tail(len(df) - 4)
                prompts = []
                
                for index, test_sample in df_testing.iterrows():
                    if test_sample['true/false']:
                        ans = "SUPPORT"
                    else:
                        ans = "CONTRADICT"
                    #find matching id in df_gt and df_testing to get the prompt 
                    abstract = df_gt[df_gt['id'] == test_sample['id']]['prompts'].values[0]
                    
                    prompt = (f"Read the claim and abstract below, then answer the question at the end:\n\n" +
                              f"Claim: {test_sample['claim']}\n" +
                              f"Abstract: {abstract}\n\n" +
                              f"Stance: {ans}\n\n" +
                              f"Question: Which of the numbered sentences support the previously selected stance? Answer with only a list of numbers separated by commas, on one line.\n\n")
                    prompts.append(prompt)

                shot_query(domain, prompts)
                result_df = pd.read_csv(export_path + domain) 
                domain_df = df  
                true_false_adjusted = domain_df['true/false'].iloc[4:].reset_index(drop=True) 
                selected_sentence_list = domain_df['selected_sentence_list'].iloc[4:].reset_index(drop=True) 
                result_df['true/false'] = true_false_adjusted
                result_df['selected_sentence_list'] = selected_sentence_list
                
                print(selected_sentence_list)
                rationale_df = pd.read_csv('gpt-3.5/data2/ground_truth_datasets/domain_rationale/indexed_rationale_gt.csv', usecols=['id', 'support', 'true/false','domain'])
                #exclude all rows that don't have the same domain
                if domain == 'health_domain.csv':
                    dom = 'Health'
                else:
                    dom = 'Humans'
                rationale_df = rationale_df[rationale_df['domain'] == dom] 
                #exclude each occurence of the first 4 unique IDs in the domain 
                unique_ids = domain_df['id'].unique()[:4]
                rationale_df = rationale_df[~rationale_df['id'].isin(unique_ids)] 
                
                combined_df = pd.concat([ rationale_df, result_df], axis=1)
                #_export_path = export_path + "combined_" + domain
                
                rationale_scorer(combined_df, domain)
                
                #df = df.apply(parse_gpt_response_rationale, axis=1)
                #combined_df.to_csv(combined_export_path, index=False)
                results = evaluateRationale_support(export_path + 'rationale_' + domain, gt_path)
                print(results)
                print("Done with domain: " + domain)
            
    
    def one_shot():
        pass
    
    def n_CoT(n):
        pass
    
    zero_shot()
    


#start_up()
prompt_generation()    