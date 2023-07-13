import os
# from dotenv import load_dotenv, find_dotenv
# from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
# from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
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
import pandas as pd
# from langchain.document_loaders import UnstructuredPDFLoader
# _ = load_dotenv(find_dotenv())

st.set_page_config(layout="wide")

hide = '''
<style>
MainMenu {visibility:hidden;}
header {visibility:hidden;}
footer {visibility:hidden;}
.css-1b9x38r {
    display: none;
    }
    
.css-1cypcdb {
    min-width: 500px;
    max-width: 500px;
}
.css-1544g2n {
    padding: 1rem 1rem 1.5rem;
}
div.block-container {
    padding-top: 0rem;
    }
</style>
'''
st.markdown(hide, unsafe_allow_html=True)


session = st.session_state
if 'transcript' not in session:
    session.transcript = []

if 'analysis' not in session:
    session.analysis = []

if 'input_disabled' not in session:
    session.input_disabled = True

if 'analyze_disabled' not in session:
    session.analyze_disabled = False

if 'institute_type' not in session:
    session.institute_type = "Full Form"

os.environ["OPENAI_API_TYPE"] ="azure"
os.environ["OPENAI_API_VERSION"] ="2023-05-15"
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://testavinx.openai.azure.com/"

# llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.1)
llm = AzureChatOpenAI(deployment_name="gpt-35-turbo",model_name="gpt-35-turbo",temperature=0)
embeddings = OpenAIEmbeddings(deployment="embedding1",
model="text-embedding-ada-002",
openai_api_base="https://testavinx.openai.azure.com/",
openai_api_type="azure",
chunk_size = 1)

template = """
You are a helpful virtual assistant of OSFI. Analyze the context and answer the question in one word "Yes" or "No" only. Remember the
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

institute_names = {"Bank of Montreal":"bmo_ar2022 (2)_index","Versa Bank":"VBAR_index","National Bank of Canada":"NATIONAL BANK OF CANADA_ 2022 Annual Report (1)_index"}

with st.sidebar:
    institute = st.selectbox(label="Institute",options=institute_names)
    bank_db = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name=institute_names[institute])

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
    
def updated_analysis(message):
    session.analysis.append(message)

def analyze():
    with st.sidebar:
        with st.spinner(f"Checking if {institute} belongs to BCAR Short Form Category"):
            session.analyze_disabled = True
            updated_analysis("The first step is to figure out whether the institute belong to BCAR Short Form, Category III or Full BCAR category.\n\nTo determine which of the above category the institute belongs to you need to answer a series of questions.")
            q1_ans = get_answer(q1)
            updated_analysis(f"1) {q1} {q1_ans}")
            session.institute_type = "Short Form"
            possibly_cat3 = False
            if q1_ans.startswith("Yes"):
                for qs in q1y_list:
                    qs_ans = get_answer(qs)
                    updated_analysis(f"{2+q1y_list.index(qs)}) {qs} {qs_ans}")    
                    if qs_ans.startswith("No"):
                        possibly_cat3 = True
                        break
            elif q1_ans.startswith("No"):
                for qs in q1n_list:
                    qs_ans = get_answer(qs)
                    updated_analysis(f"{2+q1n_list.index(qs)}) {qs} {qs_ans}")    
                    if qs_ans.startswith("No"):
                        possibly_cat3 = True
                        break
    with st.sidebar:
        with st.spinner(f"Checking if {institute} belongs to BCAR Category III Category"):
            if possibly_cat3:
                updated_analysis("Based on the answers of the above question the institude does not come under BCAR Short Form Category. We will now check if it comes under BCAR Category III")
                session.institute_type = "Category 3"
                q2_ans = get_answer(q2)
                updated_analysis(f"1) {q2} {q2_ans}")    
                if q2_ans.startswith("Yes"):
                    for qs in q2y_list:
                        qs_ans = get_answer(qs)
                        updated_analysis(f"{2+q2y_list.index(qs)}) {qs} {qs_ans}")    
                        if qs_ans.startswith("Yes"):
                            updated_analysis("Based on the answers of the above question the institude does not come under BCAR Short Form or BCAR Category II so it belongs to Full BCAR Category")
                            session.institute_type = "Full Form"
                            break
                        updated_analysis("Based on the answers of the above question the institude comes under BCAR Category III")
                else:
                    updated_analysis("Based on the answers of the above question the institude does not come under BCAR Short Form or BCAR Category II so it belongs to Full BCAR Category")
                    session.institute_type = "Full Form"
            else:
                updated_analysis("Based on the answers of the above question the institude comes under BCAR Short Form Category")
            session.input_disabled = False

    schedules = pd.read_csv("schedules.csv",delimiter="|")
    schedules = list(schedules[schedules[session.institute_type]]["Schedule Number - Schedules"])
    limited_schedules = "\n".join([f"{i+1}) {schedules[i]}\n" for i in range(len(schedules))])
    st.chat_message("assistant").write(f"According to the information provided the Institute belongs to {session.institute_type} category and thus the required schedules are limited to:\n\n{limited_schedules}")

with st.sidebar:
    analyze_button = st.button("Analyze",use_container_width=True,disabled=session.analyze_disabled,on_click=analyze)
    for message in session.analysis:
        st.write(message)                                                              


user_input = st.chat_input("Query",disabled=session.input_disabled)

bcar_db = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name="Basel Capital Adequacy Reporting (BCAR) 2023 (2)_index")
schedules_db = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name="Schedules_index")
schedules_csv_db = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name="Schedules_csv_index")

bcar_db.merge_from(schedules_db)
bcar_db.merge_from(schedules_csv_db)

chat_template = f"""
You are virtual assistant of OSFI. You have to help the user working for {institute}. Your job is to help the user file the BCAR {session.institute_type} by providing the list of schedules, 
for various types of risks such as credit risk, operation risk and market risk. Make sure to give the correct and accurate answers only.
Use the following  context (delimited by <ctx></ctx>), and the chat history (delimited by <hs></hs>) to answer the question:
"""+"""------
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
chat_prompt = PromptTemplate(input_variables=["history", "context", "question"],template=chat_template)

chat_agent = RetrievalQA.from_chain_type(llm = llm,
        chain_type='stuff', # 'stuff', 'map_reduce', 'refine', 'map_rerank'
        retriever=bcar_db.as_retriever(),
        verbose=False,
        chain_type_kwargs={
        "verbose":True,
        "prompt": chat_prompt,
        "memory": ConversationBufferMemory(
            input_key="question"),
    })


if user_input:
    session.transcript.append(["user",user_input])
    bot_output = chat_agent.run(user_input)
    session.transcript.append(["assistant",bot_output])
    for message in session.transcript:
        st.chat_message(message[0]).write(message[1])

