import time, os, openai
import pandas as pd

#from openai import OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
file_path = 'gpt-3.5/data2/claim_level_groundtruth.csv'

df = pd.read_csv(file_path)

claimIDs = []
claims = []
abstracts = []
responses = []

requests = 0

# Adjust temperature
temp = 0.0

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp, # This is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

for index, row in df.iterrows():
    claimIDs.append(row['id'])
    claims.append(row['claim'])
    abstracts.append(row['published_paper_abstract'])

print("Done with CSV read")

for index, row in df.iterrows():

    prompt = "Claim: " + claims[index] +"\nAbstract: " + abstracts[index] + "\n Question: Is the abstract relevant to the claim? Answer with one word and a number: SUPPORT if the abstract supports the claim, CONTRADICT if the abstract contradicts the claim or NEI if the abstract does not provide enough information about the claim to decide along with a number on a scale of 0-10 rate how relevant the abstract is to the claim.  \n Answer: "

    if requests>0 and requests%20 == 0:
        print("Sleeping for 70 seconds...") # Can only make 20 calls a minute, sleep time can be lower but this worked for me
        time.sleep(70)
        print("Continuing...currently on request " + str(requests))

    requests += 1

    responses.append(get_completion(prompt))

    df.at[index, 'GPT_Response_0.0'] = responses[index] # Name of column in output file

# Export to CSV
df.to_csv('gpt-3.5/data2/002_temp_consensus.csv')

print("Done with export")




