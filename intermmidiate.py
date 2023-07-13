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

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.1)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",chunk_size =1)

# template = """
# You are a helpful assistant you have to provide a detailed summary of {question} by using the context provided to you. 
# Remember to give accurate summaries. 
# Use the following context (delimited by <ctx></ctx>) for summarizing:

# <ctx>
# {context}
# </ctx>

# Answer:
# """

# prompt = PromptTemplate(input_variables=["question", "context"],template=template)


def compare_and_answer(question,docs):
    template = """You are a helpful assistant who provides all the point from the context that might be necessary to answer the following question "{question}".
    Do not try to answer the question just provide the necessary or relevant point to the question. 
    Use the following context (delimited by <ctx></ctx>) for finding out the necessary point:

    <ctx>
    {context}
    </ctx>

    Answer:"""

    prompt = PromptTemplate(input_variables=["question", "context"],template=template)
    summary = dict()
    for doc_name,doc_db in docs.items():
        agent = RetrievalQA.from_chain_type(llm = llm,
            chain_type='stuff', # 'stuff', 'map_reduce', 'refine', 'map_rerank'
            retriever=doc_db.as_retriever(),
            verbose=False,
            chain_type_kwargs={
            "verbose":True,
            "prompt": prompt,
            "memory": ConversationBufferMemory(
                input_key="question"),
        })
        summary[doc_name] = agent.run(question)

    context = "\n\n".join([f"Relevant points from {doc_name}:\n\n{doc_summary}" for doc_name,doc_summary in summary.items()])

    temp = """You are a helpful chatbot who has to answer question of a user from the institute {institute} which comes under the BCAR {institute_type} section.
    You will be given relevant points from various documents that will help you answer the user question.
    Below is a list of relevant points along with the name of the document from where thoes points are from.
    Consider all the documents provided to you and answer the question by choosing all the relevant points to the question.
    You might have to compare more points from more than one document to answer the question.

    {context}"""

    system_template = PromptTemplate(template=temp,input_variables=["institute","institute_type","context"])
    system_message_prompt = SystemMessagePromptTemplate(prompt=system_template)
    
    messages_prompt = [system_message_prompt]
    messages_prompt.append(HumanMessage(content=question))
    chat_prompt = ChatPromptTemplate.from_messages(messages_prompt)

    response = llm(chat_prompt.format_prompt(Institute=institute,institute_type=session.institute_type,context=context).to_messages()).content

    return response
    

# schedules["Summary"] = schedules["Schedule Number - Schedules"].apply(summarize)

# json_dict = schedules.to_dict()

# with open("./schedules.json","w") as f:
#     json.dump(json_dict,f,indent = 6)

# with open("./schedules.json","r") as f:
#     schedule_dict = json.load(f)

# # print(schedule_dict.keys())

# schedule_dict = [{"schedule":schedule_dict["Schedule Number - Schedules"][x]+"\n\n"+schedule_dict["Summary"][x]} for x in schedule_dict["Summary"].keys()]

# with open("./schedules_summary.json","w") as f:
#     json.dump(schedule_dict,f,indent = 3)

schedules = pd.read_csv("schedules.csv",delimiter="|")
schedules = list(schedules[schedules["Full Form"]]["Schedule Number - Schedules"])
print(schedules)
# limited_schedules = "\n".join([f"{i+1}) {schedules[i]}\n" for i in range(len(schedules))])
# st.chat_message("assistant").write(f"According to the information provided the Institute belongs to {'Full Form'} category and thus the required schedules are limited to:\n\n{limited_schedules}")