import time, os, openai, random
import pandas as pd

#api key stored as environment variable for security
openai.api_key = os.getenv('OPENAI_API_KEY')
file_path   = 'gpt-3.5/data2/ground_truth_datasets/domain/'
#file_path2 = 'gpt-3.5/data2/ground_truth_datasets/sentence_level_ground_truth-v1.csv'
export_path = 'gpt-3.5/data2/zero_shot/'

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0, # This is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def prompt_generation():
    #Given a folder of CSVs, with each CSV being a unique domain, read each CSV and generate prompts for each domain
    for file in os.listdir(file_path):
        if file.endswith(".csv") and file != "out_domain.csv":
            domain = file
            df = pd.read_csv(file_path + domain)
            #set the first four rows as the training data
            df_training = df.head(4)
            #set the rest of the rows as the testing data
            df_testing = df.tail(len(df) - 4)
            training_sample = df_training.sample(1).iloc[0]
            prompts = []    
            #generate prompts for that domain
            for index, test_sample in df_testing.iterrows():
                if training_sample['true/false']:
                    ans = "SUPPORT"
                else:  
                    ans = "CONTRADICT"

                prompt = (#f"Read the claim, abstract, and answer below, then answer the question at the end:\n"
                          #f"Claim: {training_sample['claim']}\n"
                          #f"Abstract: {training_sample['published_paper_abstract']}\n"
                          #f"Question: Does the abstract of the scientific paper support the claim?\n"
                          #f"Answer: {ans}\n\n"
                          f"Read the claim and abstract below, then answer the question at the end:\n\n"
                          f"Claim: {test_sample['claim']}\n"
                          f"Abstract: {test_sample['published_paper_abstract']}\n"
                          f"Question: Does the abstract of the scientific paper support the claim? Answer with SUPPORT or CONTRADICT.\n\n") 
                prompts.append(prompt)
            print(f"Prompts for {domain}:")
            #print the training sample
            #print(f"Training sample: {training_sample['claim']}")
            #print()
            #for p in prompts:
                #print(p)
                #print("\n---\n")
            
            shot_query(domain,prompts)
            result_df = pd.read_csv(export_path + domain)
            domain_df = pd.read_csv(file_path + domain, usecols=['claim', 'domain', 'true/false'])
            result_df['GPT_Response_1'] = result_df['GPT_Response_1'].apply(parse_response)
            true_false_adjusted = domain_df['true/false'].iloc[4:].reset_index(drop=True)
            result_df['true/false'] = true_false_adjusted
            combined_df = result_df
            combined_export_path = export_path + "combined_" + domain
            combined_df.to_csv(combined_export_path, index=False)
            eval(combined_export_path)
            print("File combined and exported: " + combined_export_path)
            print("Done with domain: " + domain)
            #END MOVE TO NEXT DOMAIN
    

def shot_query(domain,prompts):
    promptsDF = pd.DataFrame(prompts, columns=['prompts'])
    responses = []
    requests = 0
    col_name = 'GPT_Response_1'
    for index,prompt in enumerate(prompts):
        if requests>0 and requests%50 == 0:
            print("Sleeping for 70 seconds...") 
            time.sleep(70)
            print("Continuing...currently on request " + str(requests))

        requests += 1

        responses.append(get_completion(prompt))
        promptsDF.at[index, col_name] = responses[index] # Name of column in output file

    # Export to CSV(creates CSV with additional column(s))
    promptsDF.to_csv(export_path + domain)
    print("Done with export " + domain + "!\n")
    
def parse_response(response):
    if pd.isna(response):
        return None
    response = response.strip()
    if "SUPPORT" in response:
        return "SUPPORT"
    elif "CONTRADICT" in response:
        return "CONTRADICT"
    return None  

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

def eval(file_path):
    trueP = falseP = trueN = falseN = total = 0
    df = pd.read_csv(file_path)

    ground_truth = df['true/false'].astype(bool).tolist()
    labels = df['GPT_Response_1'].tolist()
    #ids = df['id'].astype(int).tolist()

    # Loop through both lists and calculate the counts
    for gt, label in zip(ground_truth, labels):
        total += 1
        if gt and label == 'SUPPORT':
            trueP += 1
            #print(f"{id} is a trueP, gt = {gt} and label = {label}")
        elif not gt and label == 'CONTRADICT': 
            trueN += 1
            #print(f"{id} is a trueN, gt = {gt} and label = {label}")
        elif not gt  and (label == 'SUPPORT' or label == 'NEI'):
            falseP += 1
            #print(f"{id} is a falseP, gt = {gt} and label = {label}")
        elif gt  and (label == 'CONTRADICT'or label == 'NEI'):
            falseN += 1
            #print(f"{id} is a falseN, gt = {gt} and label = {label}")
        print(f"TP: {trueP} TN: {trueN}\nFP: {falseP} FN: {falseN} \ntotal: {total}")

    print()
    print('Support Class Stance:\n')
    results = calculate_metrics(trueP, trueN, falseP, falseN)
    return results
    
    
#prompt_generation()
#Iterate trough 



