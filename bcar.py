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

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.1)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",chunk_size =1)

template = """
You are a helpful virtual assistant of OSFI. Analyze the context and answer the question in "Yes" or "No" only. Remember the
answer should be only "Yes" or "No". If you don't know the answer, just answer "No".
Use the following context (delimited by <ctx></ctx>) to answer the question:

------
<ctx>
{context}
</ctx>
------
{question}
Answer:
"""

prompt = PromptTemplate(input_variables=["context", "question"],template=template)

def get_answer(question):
    agent = RetrievalQA.from_chain_type(llm = llm,
        chain_type='stuff', # 'stuff', 'map_reduce', 'refine', 'map_rerank'
        retriever=bank_db.as_retriever(),
        verbose=False,
        chain_type_kwargs={
        "verbose":True,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            input_key="question"),
    })
    return agent.run(question)

session = st.session_state
if 'transcript' not in session:
    session.transcript = []

if 'input_disabled' not in session:
    session.input_disabled = True

if 'analyze_disabled' not in session:
    session.analyze_disabled = False

institute_names = {"BMO":"bmo_ar2022 (2)_index","NBC":"NATIONAL BANK OF CANADA_ 2022 Annual Report (1)_index"}

input_container = st.container()
with input_container:
    institute = st.selectbox(label="Institute",options=institute_names)
bank_db = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name=institute_names[institute])

# messages = st.container()
# user_input = st.chat_input("Query",disabled=session['input_disabled'])

q1 = f"Does {institute} have a parent company?"
q1y_list = [
    f"Is {institute}'s parent an operating company regulated by OSFI?",
    f"Has {institute}'s parent adopted an internal rating (IRB) approach to credit risk?",
    f"Is {institute} a fully- consolidated subsidiary?",
    f"Does {institute} have at least 95% of its credit risk exposures captured under the IRB approach?"
    ]
q1n_list = [
    f"Has {institute} adopted an internal rating (IRB) approach to credit risk?",
    f"Is {institute} a fully- consolidated subsidiary?",
    f"Does {institute} have at least 95% of its credit risk exposures captured under the IRB approach?"
    ]
q2 = f"Is {institute} reporting less than $10 billion in total assets?"
q2y_list = [
    f"Is {institute} reporting greater than $100 million in total loans?",
    f"Does {institute} have an interest rate or foreign exchange derivatives with a combined notional amount greater than 100% of total capital?",
    f"Does {institute} have any other types of derivative exposure?",
    f"Does {institute} have exposure to other off-balance sheet items greater than 100% of total capital?"
    ]

def analyze():
    session.analyze_disabled = True
    session.transcript.append(["assistant",q1])
    q1_ans = get_answer(q1)
    session.transcript.append(["user",q1_ans])
    st.chat_message(session.transcript[-2][0]).write(session.transcript[-2][1])
    st.chat_message(session.transcript[-1][0]).write(session.transcript[-1][1])
    institute_type = "Short Form"
    possibly_cat3 = False
    if q1_ans.startswith("Yes"):
        for qs in q1y_list:
            session.transcript.append(["assistant",qs])
            qs_ans = get_answer(qs)
            session.transcript.append(["user",qs_ans])        
            st.chat_message(session.transcript[-2][0]).write(session.transcript[-2][1])
            st.chat_message(session.transcript[-1][0]).write(session.transcript[-1][1])
            if qs_ans.startswith("No"):
                possibly_cat3 = True
                break
    elif q1_ans.startswith("No"):
        for qs in q1n_list:
            session.transcript.append(["assistant",qs])
            qs_ans = get_answer(qs)
            session.transcript.append(["user",qs_ans])
            st.chat_message(session.transcript[-2][0]).write(session.transcript[-2][1])
            st.chat_message(session.transcript[-1][0]).write(session.transcript[-1][1])
            if qs_ans.startswith("No"):
                possibly_cat3 = True
                break
    if possibly_cat3:
        institute_type = "Category III"
        session.transcript.append(["assistant",q2])
        q2_ans = get_answer(q2)
        session.transcript.append(["user",q2_ans])        
        st.chat_message(session.transcript[-2][0]).write(session.transcript[-2][1])
        st.chat_message(session.transcript[-1][0]).write(session.transcript[-1][1])
        if q2_ans.startswith("Yes"):
            for qs in q2y_list:
                session.transcript.append(["assistant",qs])
                qs_ans = get_answer(qs)
                session.transcript.append(["user",qs_ans])                
                st.chat_message(session.transcript[-2][0]).write(session.transcript[-2][1])
                st.chat_message(session.transcript[-1][0]).write(session.transcript[-1][1])
                if qs_ans.startswith("Yes"):
                    institute_type = "Full Form"
                    break
        else:
            institute_type = "Full Form"


# for message in session.transcript:
#     st.chat_message(message[0]).write(message[1])
with input_container:
    analyze_button = st.button("Analyze",use_container_width=True,disabled=session.analyze_disabled,on_click=analyze)

# if user_input:
#     output = agent.run(user_input)
#     # with relevent_docs:
#     #     st.write("\n\n\n",bcar_retriever.as_retriever().get_relevant_documents(user_input),"\n\n\n")
#     session.past.append(user_input)
#     session.generated.append(output)
# if 'generated' in session:
#     with messages:
#         for i in range(len(session['generated'])):
#             st.chat_message("user").write(session['past'][i])
#             st.chat_message("assistant").write(session["generated"][i])
