import time
import os
import openai
import random
import pandas as pd

# Set the API key for OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Parameters
 
file_path = 'gpt-3.5/data2/ground_truth_datasets/domain/'
file_path2 = 'gpt-3.5/data2/ground_truth_datasets/claim_level_ground_truth-v1.csv'


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0, # This is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def aggregate_out_domain_data(current_domain):
    out_domain_data = []
    # Iterate over all files and collect first four rows from other domains
    for file in os.listdir(file_path):
        if file.endswith(".csv") and file != current_domain and file != "out_domain.csv":
            df = pd.read_csv(file_path + file)
            # Add the first four rows to the out-domain training pool
            out_domain_data.append(df.head(4))
    # Concatenate all the dataframes collected
    return pd.concat(out_domain_data)

def prompt_generation():
    for file in os.listdir(file_path):
        if file.endswith(".csv") and file != "out_domain.csv":
            domain = file
            df = pd.read_csv(file_path + domain)

            # Aggregate out-domain data
            out_domain_training_data = aggregate_out_domain_data(domain)
            # Randomly select between 1 and num_shots samples from the out-domain training data
            selected_training = out_domain_training_data.sample(n=num_shots, random_state=42)

            df_testing = df.tail(len(df) - 4)
            prompts = [] 
            
            #generate prompts for that domain
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
            #print(f"Prompts for {domain}:")

            with open(export_path + domain + "_prompts.txt", "w") as output:
                for p in prompts:
                    output.write(p + "\n---\n")

            
            shot_query(domain, prompts)
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
            
def shot_query(domain, prompts):
    promptsDF = pd.DataFrame(prompts, columns=['prompts'])
    responses = []
    requests = 0
    for index, prompt in enumerate(prompts):
        if requests > 0 and requests % 50 == 0:
            print("Sleeping for 70 seconds...")
            time.sleep(70)
            print("Continuing...currently on request " + str(requests))
        requests += 1
        response = get_completion(prompt)
        responses.append(response)
        promptsDF.at[index, 'GPT_Response_1'] = response

    promptsDF.to_csv(export_path + domain)

def parse_response(response):
    if pd.isna(response):
        return None

    response = response.strip()
    if "SUPPORT" in response:
        return "SUPPORT"
    elif "CONTRADICT" in response:
        return "CONTRADICT"
    return None
    # Find where the conclusion starts in the response, following 'Step 4:'
    conclusion_start = response.find("Step 4:")
    if conclusion_start != -1:
        conclusion_text = response[conclusion_start:]
        # Check for keywords indicating support or contradiction
        if "supports the claim" in conclusion_text or "Conclusion: The abstract supports the claim" in conclusion_text:
            return "SUPPORT"
        elif "refutes the claim" in conclusion_text or "Conclusion: The abstract refutes the claim" in conclusion_text or "contradicts the claim" in conclusion_text or "does not support" in conclusion_text:
            return "CONTRADICT"
    #-------------------------------------------------------------------------------------------------------------------
    support_count = response.lower().count("support") + response.lower().count("supports")
    refute_count = response.lower().count("refute") + response.lower().count("refutes")
    if support_count > refute_count:
        return "SUPPORT"
    elif refute_count > support_count:
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
    with open(export_path + "_results.txt", "w") as output:
        output.write(f"TP: {trueP} TN: {trueN}\nFP: {falseP} FN: {falseN} \ntotal: {total}") 
        output.write(f"accuracy = {format(accuracy*100, '.2f')}% \nprecision = {format(precision*100,'.2f')}%\nrecall = {format(recall*100,'.2f')}%\nf1 = {format(f1*100,'.2f')}%")
        
        
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
    results = calculate_metrics(trueP, trueN, falseP, falseN)
    return results
    
    
#for num_shots in [1,2,3,4]:
#    print(f'{num_shots}-CoT\n')
#    export_path = f'gpt-3.5/data2/out/{num_shots}-CoT/'    
#    prompt_generation()
#    print("-------------------------------------------------------------------------------------------------------------------\n")
num_shots = 1
export_path = f'gpt-3.5/data2/out/one_shot/'
prompt_generation()