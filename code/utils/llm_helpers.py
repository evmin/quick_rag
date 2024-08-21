import os 

from utils.openai_helpers import *
from utils.general_helpers import *
from utils.cogsearch_rest import *


module_directory = os.path.dirname(os.path.abspath(__file__))


if os.path.exists(os.path.join(module_directory, "prompts")):
    prompt_dir = os.path.join(module_directory, "prompts")
elif os.path.exists("./prompts"):
    prompt_dir = "./prompts"
else:
    prompt_dir = "../code/prompts"


generate_tags_prompt = read_file(f'{prompt_dir}/generate_tags_prompt.txt')



def generate_tag_list(text, model = AZURE_OPENAI_MODEL, client = oai_client):
    try:
        messages = [{"role":"system", "content":generate_tags_prompt.format(text=text)}]
        result = get_chat_completion(messages, model=model, client = client) 
        return result.choices[0].message.content
    except Exception as e:
        print("Error generating tag list: ", e)
        return text


def call_ai_search(query, index_name, top=7, count=False):

    index = CogSearchRestAPI(index_name)
    results = index.search_documents(query, top=top, count=count)
    results = results['value']
    for r in results: del r['vector']
    search_results = copy.deepcopy(results)
    return search_results