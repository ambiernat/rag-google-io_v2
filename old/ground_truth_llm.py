#!/usr/bin/env python
# coding: utf-8

# In[1]:


from openai import OpenAI
import json
from pathlib import Path
from tqdm import tqdm


# #### Configs

# In[40]:


MODEL_NAME = "gpt-5-nano"
QUESTIONS_PER_DOC = 2 # development-stage

OUTPUT_PATH = Path("../data/eval/ground_truth_gpt5nano.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ## Initialise OpenAI client

# In[3]:


import getpass

try:
    API_KEY = getpass.getpass()
except Exception as error:
    print('ERROR', error)
else:
    print('API_KEY entered')


# In[4]:


client = OpenAI(api_key=API_KEY)


# ## Load documents
# (canonical source)

# In[5]:


with open("../data/canonical/all_documents.json") as f:
    documents = json.load(f)

len(documents)


# ## Prompt to generate questions

# In[6]:


def build_question_prompt(text, n_questions):
    return f"""
You are generating user search questions for a retrieval evaluation dataset.

The user has NOT seen the text below.
They are searching for information contained in it.

Generate {n_questions} DISTINCT, realistic user questions that could retrieve this text.
- Questions should vary in wording and intent
- Do NOT quote the text
- Do NOT include answers
- Do NOT number the questions
- Each question must be on a separate line

TEXT:
{text}
"""


# ## Function to call GPT-5-nano

# In[41]:


def generate_questions(text, n_questions=5):
    response = client.responses.create(
        model=MODEL_NAME,
        input=build_question_prompt(text, n_questions)
    )

    raw = response.output_text.strip()
    questions = [q.strip("- ").strip() for q in raw.split("\n") if q.strip()]

    return questions


# ## Generate ground truth dataset

# In[42]:


ground_truth = []

for doc in tqdm(documents[:5]):  # start SMALL (e.g. 20-50 docs); SCALE later
    doc_id = doc["id"]
    text = doc["text"]

    try:
        questions = generate_questions(text, QUESTIONS_PER_DOC)

        for q in questions:
            ground_truth.append({
                "query": q,
                "relevant_doc_ids": [doc_id]
            })

    except Exception as e:
        print(f"Error for doc {doc_id}: {e}")


# ## Sanity check

# In[35]:


ground_truth[:5]


# In[37]:


doc_ids=[]
for i in range(0, len(ground_truth)):
    doc_id = ground_truth[i]["relevant_doc_ids"][0]
    doc_ids.append(doc_id)


# In[38]:


set(doc_ids)


# In[39]:


doc_ids


# In[43]:


len(ground_truth)


# ## Save dataset

# In[ ]:


with open(OUTPUT_PATH, "w") as f:
    json.dump(ground_truth, f, indent=2)

len(ground_truth)


# In[ ]:




