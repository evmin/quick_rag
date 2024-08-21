

import os
import json

from env_vars import *
from utils.openai_helpers import *
from utils.llm_helpers import *


module_directory = os.path.dirname(os.path.abspath(__file__))



class Orchestrator():

    def __init__(self, index_name=KB_INDEX_NAME, topic = KB_INDEX_NAME):

        self.topic = topic
        self.index_name = index_name
        self.functions_to_call = {"search_with_rag": self.query_rag}

        system_prompt_template = read_file(os.path.join(module_directory, "prompts/orchestrator_system_prompt.txt"))
        self.system_prompt = system_prompt_template.format(topic=topic)

        self.tools_functions = json.loads(read_file(os.path.join(module_directory, "prompts/orchestrator_functions.json")))

        self.logged_messages = []

        self.messages = []
        self.messages.append({"role": "system", "content": self.system_prompt})

        
    def custom_print(self, *args, **kwargs):
        args = [str(a) for a in args]   
        logged_args = " ". join(args)
        self.logged_messages.append(logged_args)
        print(*args, **kwargs)


    def query_rag(self, search_phrase):

        context = call_ai_search(search_phrase, self.index_name)
        rag_prompt_template = read_file(os.path.join(module_directory, "prompts/orchestrator_rag_prompt.txt"))
        rag_prompt = rag_prompt_template.format(topic=self.topic, context=context, query=self.query)

        answer = ask_LLM(rag_prompt)
        return answer



    def chat(self, query):
        self.query = query ## changes every time, but saved to be used in other member methods.

        self.logged_messages = []
        self.messages.append({"role": "user", "content": query})
        answer = ask_LLM_with_functions(self.messages, self.tools_functions, self.functions_to_call, model_info = None)

        if isinstance(answer, str): 
            ## No OAI Function Call
            self.messages.append({"role": "assistant", "content": answer})        
        else:    
            ## Handle OAI Function Call
            self.messages.extend(answer)
            rets = json.loads(answer[1]['content'])
            final_answer = ""

            for k in rets:
                if 'search_with_rag' in k:
                    final_answer = final_answer + f"Answer from the RAG Database:\n" + rets[k]
                else:
                    final_answer = final_answer + f"Answer from function {k}:\n" + rets[k]

            answer = final_answer

        print(f"Final Answer:\n{answer}")
        return answer, self.logged_messages