import os
import requests

from dotenv import load_dotenv
load_dotenv()

import logging
import chainlit as cl
from chainlit import run_sync
from chainlit import make_async
import sys
sys.path.append('code')

from orchestrator import *




o = Orchestrator(KB_INDEX_NAME, KB_TOPIC)
async_chat = make_async(o.chat)


@cl.on_chat_start
async def start():
    pass



@cl.on_message
async def main(message: cl.Message):
    message_content = message.content.strip().lower()
    elements = []

    answer, app_messages = await async_chat(message_content)

    for m in app_messages:
        if m.startswith("GenFiles: "):
            files = m.replace("GenFiles: ", "").split(",")
            files = [f.strip() for f in files]
            for f in files:
                elements.append(cl.File(name=os.path.basename(f), path=f, display="inline"))
        elif m.startswith("//Here are the collected resources:"):
            elements.append(cl.Text(name=f"Log:", content=m, display="inline", language="javascript"))
        else:
            elements.append(cl.Text(name=f"Log:", content=m, display="inline"))
    
    await cl.Message(content=answer, elements = elements).send()