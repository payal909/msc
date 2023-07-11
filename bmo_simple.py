import os
# from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
from IPython.display import display, Markdown
import pandas as pd
import gradio as gr
import random
import time
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
from langchain import PromptTemplate

template = """
You are virtual assistant of OSFI.
Please follow the below mentioned Chain of Thought to answer the question:
Chain of Thought:
Thought 1:Search for context regarding all the criteria listed in BCAR for Credit Risk.

Thought 2:Search out context for under which criteria does Bank BMO fall from all the criteria listed in BCAR for Credit Risk.

Thought 3:Search for context regarding BCAR Credit Risk Criteria that Bank BMO best satisfies, and hence from that context searched find out relatable reports that it has to submit to OSFI.

Answer:

Give the response of Thought 3 as the answer. Give the answer in a pointwise format.
Use the following  context (delimited by <ctx></ctx>), and the chat history (delimited by <hs></hs>) to answer the question:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",chunk_size =1)


db_bcar = Chroma(embedding_function=embeddings,persist_directory=f"/workspaces/msc/zip_bmo emb/BCAR_Embedding")
db_bmo = Chroma(embedding_function=embeddings,persist_directory=f"/workspaces/msc/zip_bmo emb/BMO_FULL_EMBEDDING")
db_credirirb = Chroma(embedding_function=embeddings,persist_directory=f"/workspaces/msc/zip_bmo emb/IRB")
db_creditstd = Chroma(embedding_function=embeddings,persist_directory=f"/workspaces/msc/zip_bmo emb/credit_risk_standartize")
db_smsb = Chroma(embedding_function=embeddings,persist_directory=f"/workspaces/msc/zip_bmo emb/SMSB_EMBEDDING")
db_ncb = Chroma(embedding_function=embeddings,persist_directory=f"/workspaces/msc/zip_bmo emb/NBC_Embedding")

dbs = [db_bcar ,db_bmo ,db_credirirb ,db_creditstd ,db_smsb ,db_ncb ]
for db in dbs[1:]:
  db_bcar._collection.add(
  embeddings=db.get()["embeddings"],
  metadatas=db.get()["metadatas"],
  documents=db.get()["documents"],
  ids=db.get()["ids"])

  prompt = PromptTemplate(input_variables=["history", "context", "question"],template=template)

  retriever = db_bcar.as_retriever()


qa = RetrievalQA.from_chain_type(llm = llm,
    chain_type='stuff', # 'stuff', 'map_reduce', 'refine', 'map_rerank'
    retriever=retriever,
    verbose=False,
    chain_type_kwargs={
    "verbose":True,
    "prompt": prompt,
    "memory": ConversationBufferMemory(
        memory_key="history",
        input_key="question"),
})



st.title("BMO Chatbot")

if 'something' not in st.session_state:
    st.session_state.something = ''

def submit():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ''

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = []
messages = st.container()
user_input = st.text_input("Query", key="widget", on_change=submit)
if st.session_state.something:
    output = qa.run(st.session_state.something)
    st.session_state.past.append(st.session_state.something)
    st.session_state.generated.append(output)
if 'generated' in st.session_state:
    with messages:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user',avatar_style="initials",seed="U")
            message(st.session_state["generated"][i], key=str(i),avatar_style="initials",seed="B")
