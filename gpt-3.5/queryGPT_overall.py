import time, os, openai, random
import pandas as pd

#api key stored as environment variable for security
openai.api_key = os.getenv('OPENAI_API_KEY')
file_path   = 'gpt-3.5/data2/ground_truth_datasets/domain/'
num_runs = 10  # Number of repetitions per domain

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0, # This is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def prompt_generation(num_shots):
    # Ensure the export directory exists
    os.makedirs(export_path, exist_ok=True)
    
    training_data_frames = []
    testing_data_frames = []
    df_out_domain_testing = pd.read_csv('gpt-3.5/data2/ground_truth_datasets/domain/out_domain.csv')

    for file in os.listdir(file_path):
        if file.endswith(".csv") and file != "out_domain.csv":
            df = pd.read_csv(file_path + file)
            training_data_frames.append(df.head(4))
            testing_data_frames.append(df.tail(len(df) - 4))

    df_training = pd.concat(training_data_frames, ignore_index=True)
    df_testing = pd.concat(testing_data_frames, ignore_index=True)
    df_testing = pd.concat([df_testing, df_out_domain_testing], ignore_index=True)

    df_training.to_csv('gpt-3.5/data2/overall/out_training.csv', index=False)
    df_testing.to_csv('gpt-3.5/data2/overall/out_testing.csv', index=False)

    all_metrics = []
    all_results = []
    selected_training = df_training.sample(n=num_shots, random_state=42)

    for run_index in range(num_runs):        
        prompts = []
        for index, test_sample in df_testing.iterrows():
            prompt = "Read the following example(s) and answer the question at the end:\n\n"
            for _, train_sample in selected_training.iterrows():
                prompt += (f"Claim: {train_sample['claim']}\n"
                        f"Abstract: {train_sample['published_paper_abstract']}\n"
                        f"Question: Does the abstract of the scientific paper support the claim?\n\n"
                        f"Answer: {train_sample['rationale']}\n\n")
            prompt += (f"Read the claim and abstract and answer the question by mimicking the process previously outlined.\n\n"
                    f"Claim: {test_sample['claim']}\n"
                    f"Abstract: {test_sample['published_paper_abstract']}\n"
                    f"Question: Does the abstract of the scientific paper support the claim?\n\n")
            prompts.append(prompt)
        
        result_df = shot_query("overall", prompts, run_index)
        result_df['true/false'] = df['true/false'].iloc[4:].reset_index(drop=True)
        all_results.append(result_df)
        all_metrics.append(calculate_metrics(result_df))
    
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_export_path = export_path + "combined_overall.csv"
    combined_df.to_csv(combined_export_path, index=False)
    print(combined_export_path + '\n')
    
    # Print the results of each run from all_metrics
    for i, m in enumerate(all_metrics):
        print(f"Run {i+1}: Precision={m['precision']:.2f}, Recall={m['recall']:.2f}, F1={m['f1']:.2f}")

    avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
    avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics)
    print(f"Averages: Precision={avg_precision:.2f}, Recall={avg_recall:.2f}, F1={avg_f1:.2f}")
    
def prompt_generation_2 (num_shots):
    # Ensure the export directory exists
    os.makedirs(export_path, exist_ok=True)
    
    training_data_frames = []
    testing_data_frames = []
    df_out_domain_testing = pd.read_csv('gpt-3.5/data2/ground_truth_datasets/domain/out_domain.csv')

    for file in os.listdir(file_path):
        if file.endswith(".csv") and file != "out_domain.csv":
            df = pd.read_csv(file_path + file)
            training_data_frames.append(df.head(4))
            testing_data_frames.append(df.tail(len(df) - 4))

    df_training = pd.concat(training_data_frames, ignore_index=True)
    df_testing = pd.concat(testing_data_frames, ignore_index=True)
    df_testing = pd.concat([df_testing, df_out_domain_testing], ignore_index=True)

    df_training.to_csv('gpt-3.5/data2/overall/out_training.csv', index=False)
    df_testing.to_csv('gpt-3.5/data2/overall/out_testing.csv', index=False)

    all_metrics = []
    all_results = []
    #selected_training = df_training.sample(n=num_shots, random_state=42)

    for run_index in range(num_runs):        
        prompts = []
        for index, test_sample in df_testing.iterrows():
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
        
        result_df = shot_query("overall", prompts, run_index)
        result_df['true/false'] = df['true/false'].iloc[4:].reset_index(drop=True)
        all_results.append(result_df)
        all_metrics.append(calculate_metrics(result_df))
    
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_export_path = export_path + "combined_overall.csv"
    combined_df.to_csv(combined_export_path, index=False)
    print(combined_export_path + '\n')
    
    # Print the results of each run from all_metrics
    for i, m in enumerate(all_metrics):
        print(f"Run {i+1}: Precision={m['precision']:.2f}, Recall={m['recall']:.2f}, F1={m['f1']:.2f}")

    avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
    avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics)
    print(f"Averages: Precision={avg_precision:.2f}, Recall={avg_recall:.2f}, F1={avg_f1:.2f}")

def prompt_generation_3 (num_shots):
    # Ensure the export directory exists
    os.makedirs(export_path, exist_ok=True)
    
    training_data_frames = []
    testing_data_frames = []
    df_out_domain_testing = pd.read_csv('gpt-3.5/data2/ground_truth_datasets/domain/out_domain.csv')

    for file in os.listdir(file_path):
        if file.endswith(".csv") and file != "out_domain.csv":
            df = pd.read_csv(file_path + file)
            training_data_frames.append(df.head(4))
            testing_data_frames.append(df.tail(len(df) - 4))

    df_training = pd.concat(training_data_frames, ignore_index=True)
    df_testing = pd.concat(testing_data_frames, ignore_index=True)
    df_testing = pd.concat([df_testing, df_out_domain_testing], ignore_index=True)

    df_training.to_csv('gpt-3.5/data2/overall/out_training.csv', index=False)
    df_testing.to_csv('gpt-3.5/data2/overall/out_testing.csv', index=False)

    all_metrics = []
    all_results = []
    selected_training = df_training.sample(n=num_shots, random_state=42)

    for run_index in range(num_runs):        
        prompts = []
        for index, test_sample in df_testing.iterrows():
            if selected_training['true/false'].iloc[0] == True:
                ans = "SUPPORT"
            else:  
                ans = "CONTRADICT"
            prompt = (f"Read the claim, abstract, and answer below, then answer the question at the end:\n"
                    f"Claim: {selected_training['claim']}\n"
                    f"Abstract: {selected_training['published_paper_abstract']}\n"
                    f"Question: Does the abstract of the scientific paper support the claim?\n"
                    f"Answer: {ans}\n\n"
                    f"Read the claim and abstract below, then answer the question at the end:\n\n"
                    f"Claim: {test_sample['claim']}\n"
                    f"Abstract: {test_sample['published_paper_abstract']}\n"
                    f"Question: Does the abstract of the scientific paper support the claim? Answer with SUPPORT or CONTRADICT.\n\n") 
            prompts.append(prompt)
        
        result_df = shot_query("overall", prompts, run_index)
        result_df['true/false'] = df['true/false'].iloc[4:].reset_index(drop=True)
        all_results.append(result_df)
        all_metrics.append(calculate_metrics(result_df))
    
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_export_path = export_path + "combined_overall.csv"
    combined_df.to_csv(combined_export_path, index=False)
    print(combined_export_path + '\n')

    # Print the results of each run from all_metrics
    for i, m in enumerate(all_metrics):
        print(f"Run {i+1}: Precision={m['precision']:.2f}, Recall={m['recall']:.2f}, F1={m['f1']:.2f}")

    avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
    avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics)
    print(f"Averages: Precision={avg_precision:.2f}, Recall={avg_recall:.2f}, F1={avg_f1:.2f}")
            
    
def shot_query(domain, prompts, run_index):
    promptsDF = pd.DataFrame(prompts, columns=['prompts'])
    responses = []
    requests = 0
    for index, prompt in enumerate(prompts):
        if requests > 0 and requests % 70 == 0:
            print("Sleeping for 70 seconds...")
            time.sleep(70)
            print("Continuing...currently on request " + str(requests))
        
        responses.append(get_completion(prompt))
        requests += 1
    promptsDF['GPT_Response_1'] = responses
    promptsDF['GPT_Response_1'] = promptsDF['GPT_Response_1'].apply(parse_response_1) 
    
    # Saving each run's result in separate CSVs
    promptsDF.to_csv(f"{export_path}{domain}_run_{run_index}.csv")
    print(f"Done with export {domain} run {run_index}!\n")
    return promptsDF

def parse_response_1(response):
    if pd.isna(response):
        return None
    response = response.strip()
    if "SUPPORT" in response:
        return "SUPPORT"
    elif "CONTRADICT" in response:
        return "CONTRADICT"
    return None

def parse_response(response):
    if pd.isna(response):
        return None

    response = response.strip()
    # Find where the conclusion starts in the response, following 'Step 4:'
    conclusion_start = response.find("Step 4:")
    if conclusion_start != -1:
        conclusion_text = response[conclusion_start:]
        # Check for keywords indicating support or contradiction
        if "supports the claim" in conclusion_text or "Conclusion: The abstract supports the claim" in conclusion_text:
            return "SUPPORT"
        elif "refutes the claim" in conclusion_text or "Conclusion: The abstract refutes the claim" in conclusion_text:
            return "CONTRADICT"
    #-------------------------------------------------------------------------------------------------------------------
    support_count = response.lower().count("support") + response.lower().count("supports")
    refute_count = response.lower().count("refute") + response.lower().count("refutes")
    if support_count > refute_count:
        return "SUPPORT"
    elif refute_count > support_count:
        return "CONTRADICT"
    
    return None
    

def calculate_metrics(df):
    trueP = falseP = trueN = falseN = total = 0
    #df = pd.read_csv(df)
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
        elif not gt  and (label == 'SUPPORT' or (label == 'NEI' or label == None)):
            falseP += 1
            #print(f"{id} is a falseP, gt = {gt} and label = {label}")
        elif gt  and (label == 'CONTRADICT'or (label == 'NEI' or label == None)):
            falseN += 1
            #print(f"{id} is a falseN, gt = {gt} and label = {label}")
        #print(f"TP: {trueP} TN: {trueN}\nFP: {falseP} FN: {falseN} \ntotal: {total}")

    precision = trueP / (trueP + falseP) if (trueP + falseP) > 0 else 0
    recall = trueP / (trueP + falseN) if (trueP + falseN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    #write the results to a text file 
    
    return {'precision': precision, 'recall': recall, 'f1': f1}    



#export_path = f'gpt-3.5/data2/overall/zero_shot/'
#prompt_generation_2(0)
#export_path = f'gpt-3.5/data2/overall/one_shot/'
#prompt_generation_3(1)
#for num_shots in [0,1]:
#    print(f'{num_shots}_shot\n')
#    export_path = f'gpt-3.5/data2/overall/{num_shots}_shot/'    
#    prompt_generation(num_shots)
#    print("-------------------------------------------------------------------------------------------------------------------\n")
#for num_shots in [1,2,3,4]:
#    print(f'{num_shots}-CoT\n')
#    export_path = f'gpt-3.5/data2/overall/{num_shots}-CoT/'    
#    prompt_generation(num_shots)
#   print("-------------------------------------------------------------------------------------------------------------------\n")


#fixes the issue with calculations and csv file combinations
p = '4-CoT'
in_path = f'gpt-3.5/data2/overall/{p}/'
gt_path = 'gpt-3.5/data2/overall/out_testing.csv'
export_path = f'gpt-3.5/data2/overall/{p}/combined_overall_fixed.csv'
df_gt = pd.read_csv(gt_path)
dataframes = []
for file in os.listdir(in_path):
    if file.endswith(".csv") and file != "combined_overall.csv":
        df = pd.read_csv(in_path + file)
        #add the 'true/false' column to the df no offsets or anything 
        df['true/false'] = df_gt['true/false'].iloc[:len(df)].reset_index(drop=True)
        dataframes.append(df)

all_metrics = []
for df in dataframes:
    all_metrics.append(calculate_metrics(df))

print(f'{p}\n')
for i, m in enumerate(all_metrics):
    print(f"Run {i+1}: Precision={m['precision']:.2f}, Recall={m['recall']:.2f}, F1={m['f1']:.2f}")

avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics)
print(f"Averages: Precision={avg_precision:.2f}, Recall={avg_recall:.2f}, F1={avg_f1:.2f}")

# Combine all dataframes into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv(export_path, index=False)
        
        
        
        

        
        
    








