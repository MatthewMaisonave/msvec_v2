import openai, time, os
import pandas as pd

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp, # This is the degree of randomness of the model's output
    )
    return response.choices[0].message.content


inFile = 'gpt-3.5/data2/ground_truth_datasets/sentence_level_ground_truth-v1.csv'
inFile_unindexed = 'gpt-3.5/data2/ground_truth_datasets/claim_level_ground_truth-v1.csv'
outFile = 'gpt-3.5/data2/indexed_rationale_responses.csv'
outFile_unindexed = 'gpt-3.5/data2/stance_query_responses.csv'


#api key stored as environment variable for security
openai.api_key = os.getenv('OPENAI_API_KEY')

requests = 0

# Adjust GPT temperature
temp = 0.0
i = 0
claimIDs = [] 
abstracts = []
abstractsNoBrackets = []
prompts = []
question = "Question(T1): Read the claim and abstract. Is the abstract relevant to the claim? Answer with one word and one number: SUPPORT if the abstract supports the claim, CONTRADICT if the abstract contradicts the claim or NEI if the abstract does not provide enough information about the claim to decide, only pick one; include one number on a scale of 0-10 only to rate how relevant the abstract is to the claim. Additionally, state the level of confidence in your answer with a number in the range 0-1 after the word 'Confidence:'. Separate each value by a single space. Answer this entire question on its own, single line starting with 'T1:', everything should be on one line, including the confidence.\nQuestion(T2): Which of the numbered sentences support the stance you previously selected? Answer with only a list of numbers separated by commas. Additionally, state the level of confidence in your answer with a single number in the range 0-1 after the word 'Confidence:', separate this value from the list of numbers with a single space. Answer this entire question, on its own, single line starting with 'T2:', everything should be on one line, including the confidence.\n"
responses = []



df = pd.read_csv(inFile)
df2 = pd.read_csv(inFile_unindexed, usecols=['id', 'title', 'claim', 'Type', 'url', 'domain', 'news_date', 'true/false', 'published_paper_urls', 'published_paper_title', 'published_paper_authors', 'published_paper_abstract', 'published_paper_venue', 'published_paper_year'])

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

    if(nextID != currentID):
        prompts.append("Claim: " + str(row['claim']) + "\n" + question + abstractsNoBrackets)
        abstracts = []
        abstractsNoBrackets = []
        i = 0

promptsDF = pd.DataFrame(prompts, columns=['prompts'])
promptsDF2 = pd.DataFrame(prompts, columns=['prompts'])
promptsDF = pd.concat([df, promptsDF], axis=1)
promptsDF2 = pd.concat([df2, promptsDF2], axis=1)

print('Done with prompt generation starting query\n')


for pos, p in enumerate(prompts):

    if requests>0 and requests%50 == 0:

        # Can make 50 calls a minute, adjust based on model
        print("\nSleeping for 60 seconds...\n")
        time.sleep(60)
        print("Continuing...currently on request " + str(requests))

    attempts = 0
    success = False

    # Kept getting ServiceUnavailableError without try block
    while attempts < 10 and not success:

        try:
            responses.append(get_completion(p))
            success = True

        except openai.ServiceUnavailableError as e:
            attempts += 1
            if attempts == 10:
                break

    # Name of column in output file
    promptsDF.at[pos, 'GPT_Response_1'] = responses[pos] 
    promptsDF2.at[pos, 'GPT_Response_1'] = responses[pos]
    requests += 1

promptsDF.to_csv(outFile, index=False)
promptsDF2.to_csv(outFile_unindexed, index=False)
print('\nDone with export!')
