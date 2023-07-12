import os
# from dotenv import load_dotenv, find_dotenv
# from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
# from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
# from langchain.llms import AzureOpenAI
# from langchain.document_loaders import DirectoryLoader,PyPDFLoader
# from langchain.document_loaders import UnstructuredExcelLoader
# from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
# from IPython.display import display, Markdown
# import pandas as pd
# import gradio as gr
# from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
# from langchain.vectorstores import Chroma
# from langchain.agents.tools import Tool
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
# from langchain import OpenAI, VectorDBQA
# from langchain.chains.router import MultiRetrievalQAChain
import streamlit as st
# from langchain.document_loaders import UnstructuredPDFLoader
# _ = load_dotenv(find_dotenv())
import pandas as pd
import json

schedules = pd.read_csv("schedules.csv",delimiter="|")


os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.1)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",chunk_size =1)

template = """
You are a helpful assistant you have to provide a detailed summary of {question} by using the context provided to you. 
Remember to give accurate summaries. 
Use the following context (delimited by <ctx></ctx>) for summarizing:

<ctx>
{context}
</ctx>

Answer:
"""

prompt = PromptTemplate(input_variables=["question", "context"],template=template)

bcar_db = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name="Basel Capital Adequacy Reporting (BCAR) 2023 (2)_index")

def summarize(schedules):
    agent = RetrievalQA.from_chain_type(llm = llm,
        chain_type='stuff', # 'stuff', 'map_reduce', 'refine', 'map_rerank'
        retriever=bcar_db.as_retriever(),
        verbose=False,
        chain_type_kwargs={
        "verbose":True,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            input_key="question"),
    })
    return agent.run(schedules)

schedules["Summary"] = schedules["Schedule Number - Schedules"].apply(summarize)

json_dict = schedules.to_dict()

with open("./schedules.json","w") as f:
    json.dump(json_dict,f,indent = 6)

with open("./schedules.json","r") as f:
    schedule_dict = json.load(f)

# print(schedule_dict.keys())

schedule_dict = [{"schedule":schedule_dict["Schedule Number - Schedules"][x],"summary":schedule_dict["Summary"][x]} for x in schedule_dict["Summary"].keys()]

with open("./schedules_summary.json","w") as f:
    json.dump(schedule_dict,f,indent = 3)