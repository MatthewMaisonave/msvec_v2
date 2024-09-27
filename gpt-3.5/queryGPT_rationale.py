import time, os, openai, re
import pandas as pd

#api key stored as environment variable for security
openai.api_key = os.getenv('OPENAI_API_KEY')
file_path   = 'gpt-3.5/data2/ground_truth_datasets/domain/'
gt_path = 'gpt-3.5/data2/ground_truth_datasets/sentence_level_ground_truth-v1_domain.csv'
#export_path = 'gpt-3.5/data2/rationale/one_shot/'

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0, # This is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def evaluateRationale_support(result_df, dom, export_path):
    trueP = 0
    falseP = 0
    trueN = 0
    falseN = 0
    
    for index, row in result_df.iterrows():
        response_sentences = parse_gpt_response_rationale(row['GPT_Response_1'])
        ground_truth = parse_ground_truth(row['selected_sentence_list'])
        
        for idx, support in ground_truth.items():
            if idx in response_sentences and support == 1:
                trueP += 1
            elif idx in response_sentences and support == 0:
                falseP += 1
            elif idx not in response_sentences and support == 1:
                falseN += 1
            elif idx not in response_sentences and support == 0:
                trueN += 1

    return calculate_metrics(trueP, falseP, trueN, falseN, dom, export_path)

def calculate_metrics(trueP, trueN, falseP, falseN, dom, export_path):
    total = trueP + trueN + falseP + falseN
    accuracy = (trueP + trueN) / total
    precision = trueP / (trueP + falseP) if (trueP + falseP) != 0 else 0
    recall = trueP / (trueP + falseN) if (trueP + falseN) != 0 else 0
    f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0 

    print(f"TP: {trueP} TN: {trueN}\nFP: {falseP} FN: {falseN} \ntotal: {total}")
    print(f"accuracy = {format(accuracy*100, '.2f')}% \nprecision = {format(precision*100,'.2f')}%\nrecall = {format(recall*100,'.2f')}%\nf1 = {format(f1*100,'.2f')}%")
    results = {'accuracy': format(accuracy*100, '.2f'), 'precision':format(precision*100,'.2f'), 'recall': format(recall*100,'.2f'), 'f1':format(f1*100,'.2f')}
    with open(export_path + dom + "_results.txt", "w") as output:
        output.write(f"TP: {trueP} TN: {trueN}\nFP: {falseP} FN: {falseN} \ntotal: {total}\n") 
        output.write(f"accuracy = {format(accuracy*100, '.2f')}% \nprecision = {format(precision*100,'.2f')}%\nrecall = {format(recall*100,'.2f')}%\nf1 = {format(f1*100,'.2f')}%")
    return results


    
    
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
                    sentences_indices.append({local_index : 1})
                else:
                    sentences_indices.append({local_index : 0})
                local_index += 1
            previous_id = i_row['id']

        # Store the list of indices for this particular ID
        selected_sentence_lists.append(sentences_indices)

    # Assign the collected indices back to the unindexed dataframe
    unindexed_df['selected_sentence_list'] = selected_sentence_lists
    return unindexed_df

def parse_gpt_response_rationale(response):
    #create a case to handle a response with "Answer: 1, 2, 3" format
    if "Answer:" in response:
        response = response.split("Answer:")[1]
        
        
    pattern = re.compile(r'(\d+)')
    match = pattern.findall(response)
    if match:
        selected_sentences = [int(num) for num in match]
        return selected_sentences
    return []

def parse_ground_truth(selected_sentence_list):
    ground_truth = eval(selected_sentence_list)
    gt_dict = {int(k): int(v) for d in ground_truth for k, v in d.items()}
    return gt_dict

def parse_gt_2(ground_truth_string):
    # Remove square brackets
    cleaned_string = ground_truth_string.strip('[]')
    # Use regex to find all occurrences of {index: 1}
    pattern = re.compile(r'\{(\d+): 1\}')
    matches = pattern.findall(cleaned_string)
    # Join matches with commas to create the desired string format
    result_string = ', '.join(matches)
    return result_string
    



def start_up():
    inFile = 'gpt-3.5/data2/ground_truth_datasets/sentence_level_ground_truth-v1_domain.csv'
    inFile_unindexed = 'gpt-3.5/data2/ground_truth_datasets/claim_level_ground_truth-v1_for_rationale.csv'
    inDir_domain = 'gpt-3.5/data2/ground_truth_datasets/domain_rationale/domain/'    
    outDir_domain = 'gpt-3.5/data2/ground_truth_datasets/domain_rationale/'
    
    def generate_gt_rationales():
        df = pd.read_csv(inFile)
        df2 = pd.read_csv(inFile_unindexed, usecols=['id', 'title', 'claim', 'Type', 'url', 'domain', 'true/false', 'published_paper_urls', 'published_paper_title', 'published_paper_abstract'])
        
        df2 = attach_supporting_sentences(df2, df)
        df2.to_csv(outDir_domain + 'unindexed_rationale_gt.csv', index=False)
        df.to_csv(outDir_domain + 'indexed_rationale_gt.csv', index=False)
        
        #for each file in inDir_domain, add the new selected_sentence_list column from the unindexed_rationale_gt 
        for file in os.listdir(inDir_domain): 
            if file.endswith(".csv"):
                domain = file
                df_domain = pd.read_csv(inDir_domain + domain)
                
                # Initialize the 'selected_sentence_list' column in df_domain
                df_domain['selected_sentence_list'] = None
                
                # Iterate through each row in df_domain and find the matching row in df2
                for index, row in df_domain.iterrows():
                    id = row['id']
                    df2_row = df2[df2['id'] == id]
                    if not df2_row.empty:
                        df_domain.at[index, 'selected_sentence_list'] = df2_row['selected_sentence_list'].values[0]
                        
                columns_to_drop = ['Type','Answers', 'Answers.1', 'Answers.2', 'Prompt']
                df_domain.drop(columns=columns_to_drop, inplace=True, errors='ignore')
                df_domain.to_csv(inDir_domain + domain, index=False)
                       
    generate_gt_rationales()
    
    
  
    
    
    
def prompt_generation():
    file_path = 'gpt-3.5/data2/ground_truth_datasets/domain_rationale/domain/'
    #export_path = 'gpt-3.5/data2/rationale/one_shot/'
    num_runs = 1

    def shot_query(domain,prompts, export_path):
        promptsDF = pd.DataFrame(prompts, columns=['prompts'])
        responses = []
        requests = 0
        col_name = 'GPT_Response_1'
        for index,prompt in enumerate(prompts):
            if requests>0 and requests%70 == 0:
                print("Sleeping for 70 seconds...") 
                time.sleep(65)
                print("Continuing...currently on request " + str(requests))

            requests += 1

            responses.append(get_completion(prompt))
            promptsDF.at[index, col_name] = responses[index] # Name of column in output file 
        
        # Export to CSV(creates CSV with additional column(s))
        promptsDF.to_csv(export_path + domain)
        print("Done with export " + domain + "!\n")
        #return promptsDF
    
    def zero_shot(num_shots=0):
        export_path = 'gpt-3.5/data2/rationale/zero_shot/'
        for file in os.listdir(file_path):
            if file.endswith(".csv") and (file in ['health_domain.csv','humans_domain.csv']):
                domain = file
                f_name = file_path + domain
                df_full = pd.read_csv(f_name)
                #first 4 rows are training data,the rest are testing data
                df_training = df_full.head(4)
                training_ids = df_training['id'].tolist()
                selected_training = df_training.sample(n=num_shots, random_state=42)
                df_testing = df_full.tail(len(df_full) - 4)
                
                #prompt generation
                f_idx = f_name.replace('.csv','_idx.csv')
                df_domain_indexed = pd.read_csv(f_idx)
                i = 0
                claimIDs = [] 
                abstracts = []
                abstractsNoBrackets = []
                indexed_abstracts = []
                for index, row in df_domain_indexed.iterrows():
                    # Used to compare next id in index during iteration
                    claimIDs.append(row['id'])

                for index, row in df_domain_indexed.iterrows():
                    currentID = row['id']
                    if(index <= (len(claimIDs) - 2)):
                        nextID = claimIDs[index + 1]

                    if(index == (len(claimIDs) - 1)):
                        nextID = 0

                    abstracts.append(str(i)+". "+row['published_paper_abstract'] + "\n")
                    abstractsNoBrackets = ''.join(abstracts)
                    
                    i += 1

                    if(nextID != currentID):
                        indexed_abstracts.append(abstractsNoBrackets)
                        abstracts = []
                        abstractsNoBrackets = []
                        i = 0
                
                
                prompts = []
                #exclude the first 4 rows of the indexed_abstracts
                indexed_abstracts = indexed_abstracts[4:]
                i = 0
                for index, test_sample in df_testing.iterrows():
                    if test_sample['true/false']:
                        ans = "SUPPORT"
                    else:
                        ans = "CONTRADICT"
                        
                    prompt = (f"Read the claim and abstract below, then answer the question at the end:\n\n" +
                              f"Claim: {test_sample['claim']}\n" +
                              f"Abstract: \n{indexed_abstracts[i]}\n\n" +
                              f"Stance: {ans}\n\n" +
                              f"Question: Which of the numbered sentences support the previously selected stance? Answer with only a list of numbers separated by commas, on one line.\n\n")
                    prompts.append(prompt)
                    i += 1

                shot_query(domain, prompts, export_path)
                result_df = pd.read_csv(export_path + domain) 
                domain_df = df_full
                selected_sentence_list = domain_df['selected_sentence_list'].iloc[4:].reset_index(drop=True) 
                result_df['selected_sentence_list'] = selected_sentence_list
                result_df.to_csv(export_path + "combined_" + domain, index=False)
                dom = domain.replace('.csv','')
                results = evaluateRationale_support(result_df, dom, export_path)
                print(results)
                
    
    def one_shot(num_shots=1):
        export_path = 'gpt-3.5/data2/rationale/one_shot/'
        for file in os.listdir(file_path):
            if file.endswith(".csv") and (file in ['health_domain.csv','humans_domain.csv']):
                domain = file
                f_name = file_path + domain
                df_full = pd.read_csv(f_name)
                #first 4 rows are training data,the rest are testing data
                df_training = df_full.head(4)
                training_ids = df_training['id'].tolist()
                selected_training = df_training.sample(n=num_shots, random_state=42)
                selected_training_ids = selected_training['id'].tolist()
                df_testing = df_full.tail(len(df_full) - 4)
                
                #prompt generation
                f_idx = f_name.replace('.csv','_idx.csv')
                df_domain_indexed = pd.read_csv(f_idx)
                i = 0
                claimIDs = [] 
                abstracts = []
                abstractsNoBrackets = []
                indexed_abstracts = []
                training_abstracts = []
                for index, row in df_domain_indexed.iterrows():
                    # Used to compare next id in index during iteration
                    claimIDs.append(row['id'])

                for index, row in df_domain_indexed.iterrows():
                    currentID = row['id']
                    if(index <= (len(claimIDs) - 2)):
                        nextID = claimIDs[index + 1]

                    if(index == (len(claimIDs) - 1)):
                        nextID = 0

                    abstracts.append(str(i)+". "+row['published_paper_abstract'] + "\n")
                    abstractsNoBrackets = ''.join(abstracts)
                    
                    i += 1

                    if(nextID != currentID):
                        if (currentID in selected_training_ids):
                            training_abstracts.append(abstractsNoBrackets)
                        indexed_abstracts.append(abstractsNoBrackets)
                        abstracts = []
                        abstractsNoBrackets = []
                        i = 0
                
                
                prompts = []
                #exclude the first 4 rows of the indexed_abstracts
                indexed_abstracts = indexed_abstracts[4:]
                i = 0
                #for each selected training sample, if true, SUPPORT, else CONTRADICT
                
                    
                for index, test_sample in df_testing.iterrows():
                    if test_sample['true/false']:
                        ans = "SUPPORT"
                    else:
                        ans = "CONTRADICT"
                    
                    for _, train_sample in selected_training.iterrows():
                        if train_sample['true/false'] == True:
                            t_ans = "SUPPORT"
                        else:
                            t_ans = "CONTRADICT"
                            
                        prompt = (f"Read the claim and abstract below, then answer the question at the end:\n\n" +
                              f"Claim: {train_sample['claim']}\n" +
                              f"Abstract: \n{training_abstracts[0]}\n\n" +
                              f"Stance: {t_ans}\n\n" +
                              f"Question: Which of the numbered sentences support the previously selected stance? Answer with only a list of numbers separated by commas, on one line.\n\n" + 
                              f"Answer:{parse_gt_2(train_sample['selected_sentence_list'])}\n\n")
                            
                    prompt += (f"Read the claim and abstract below, then answer the question at the end:\n\n" +
                              f"Claim: {test_sample['claim']}\n" +
                              f"Abstract: \n{indexed_abstracts[i]}\n\n" +
                              f"Stance: {ans}\n\n" +
                              f"Question: Which of the numbered sentences support the previously selected stance? Answer with only a list of numbers separated by commas, on one line.\n\n")
                    prompts.append(prompt)
                    i += 1
                    
                print(f"Training sample(s):" + str(selected_training['claim'].tolist()) + "\n")

                shot_query(domain, prompts, export_path)
                result_df = pd.read_csv(export_path + domain) 
                domain_df = df_full
                selected_sentence_list = domain_df['selected_sentence_list'].iloc[4:].reset_index(drop=True) 
                result_df['selected_sentence_list'] = selected_sentence_list
                result_df.to_csv(export_path + "combined_" + domain, index=False)
                dom = domain.replace('.csv','')
                results = evaluateRationale_support(result_df, dom, export_path)
                print(results)
    
    def n_CoT(num_shots):
        export_path = f'gpt-3.5/data2/rationale/{num_shots}-CoT/'
        for file in os.listdir(file_path):
            if file.endswith(".csv") and (file in ['health_domain.csv','humans_domain.csv']):
                domain = file
                f_name = file_path + domain
                df_full = pd.read_csv(f_name)
                #first 4 rows are training data,the rest are testing data
                df_training = df_full.head(4)
                training_ids = df_training['id'].tolist()
                selected_training = df_training.sample(n=num_shots, random_state=42)
                selected_training_ids = selected_training['id'].tolist()
                df_testing = df_full.tail(len(df_full) - 4)
                
                #prompt generation
                f_idx = f_name.replace('.csv','_idx.csv')
                df_domain_indexed = pd.read_csv(f_idx)
                i = 0
                claimIDs = [] 
                abstracts = []
                abstractsNoBrackets = []
                indexed_abstracts = []
                training_abstracts = []
                for index, row in df_domain_indexed.iterrows():
                    # Used to compare next id in index during iteration
                    claimIDs.append(row['id'])

                for index, row in df_domain_indexed.iterrows():
                    currentID = row['id']
                    if(index <= (len(claimIDs) - 2)):
                        nextID = claimIDs[index + 1]

                    if(index == (len(claimIDs) - 1)):
                        nextID = 0

                    abstracts.append(str(i)+". "+row['published_paper_abstract'] + "\n")
                    abstractsNoBrackets = ''.join(abstracts)
                    
                    i += 1

                    if(nextID != currentID):
                        if (currentID in selected_training_ids):
                            training_abstracts.append(abstractsNoBrackets)
                        indexed_abstracts.append(abstractsNoBrackets)
                        abstracts = []
                        abstractsNoBrackets = []
                        i = 0
                
                
                prompts = []
                #exclude the first 4 rows of the indexed_abstracts
                indexed_abstracts = indexed_abstracts[4:]
                i = 0
                #for each selected training sample, if true, SUPPORT, else CONTRADICT
                
                    
                for index, test_sample in df_testing.iterrows():
                    prompt = ""
                    if test_sample['true/false']:
                        ans = "SUPPORT"
                    else:
                        ans = "CONTRADICT"
                    j = 0
                    for _, train_sample in selected_training.iterrows():
                        if train_sample['true/false'] == True:
                            t_ans = "SUPPORT"
                        else:
                            t_ans = "CONTRADICT"
                            
                        prompt += (f"Read the following example(s) and answer the question at the end:\n\n" +
                              f"Claim: {train_sample['claim']}\n" +
                              f"Abstract: \n{training_abstracts[j]}\n\n" +
                              f"Stance: {t_ans}\n\n" +
                              f"Question: Which of the numbered sentences support the previously selected stance? Answer with only a list of numbers separated by commas, on one line.\n\n" + 
                              f"Answer: {parse_gt_2(train_sample['selected_sentence_list'])}\n\n"
                              f"Rationale: \n{train_sample['rationale']}\n\n" )
                        j += 1
                            
                    prompt += (f"Read the claim and abstract, then answer the question below by mimicking the process previously outlined.\n\n" +
                              f"Claim: {test_sample['claim']}\n" +
                              f"Abstract: \n{indexed_abstracts[i]}\n\n" +
                              f"Stance: {ans}\n\n" +
                              f"Question: Which of the numbered sentences support the previously selected stance? Answer with only a list of numbers separated by commas, on one line.\n\n")
                    prompts.append(prompt)
                    i += 1
                    
                print(f"Training sample(s):" + str(selected_training['claim'].tolist()) + "\n")

                shot_query(domain, prompts, export_path)
                result_df = pd.read_csv(export_path + domain) 
                domain_df = df_full
                selected_sentence_list = domain_df['selected_sentence_list'].iloc[4:].reset_index(drop=True) 
                result_df['selected_sentence_list'] = selected_sentence_list
                result_df.to_csv(export_path + "combined_" + domain, index=False)
                dom = domain.replace('.csv','')
                results = evaluateRationale_support(result_df, dom, export_path)
                print(results)
    
        
        
        
    print("\n0-shot\n")
    zero_shot()
    print('--------------------------------------------------')
    print("\n1-shot\n")
    one_shot()
    print('--------------------------------------------------')
    for i in [1,2,3,4]:
        print(f"\n{i}-CoT\n")
        n_CoT(i)
        print('--------------------------------------------------')

    


#start_up()
prompt_generation()  

