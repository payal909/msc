from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.document_loaders import PyPDFLoader
from tqdm import tqdm
from langchain.chat_models import ChatAnthropic
import os
import streamlit as st

session = st.session_state
if 'transcript' not in session:
    session.transcript = []

# os.environ["ANTHROPIC_API_KEY"] = "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"
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
claude_models = ["claude-instant-1","claude-2"]
llm = ChatAnthropic(model=claude_models[1],temperature= 0)

institute = "Bank of Montreal (BMO)"
institute_type = "Full Form"
institute_data_paths = {
    "Bank of Montreal (BMO)"            :   "./data/bmo_ar2022 (2).pdf",
    "Versa Bank"                        :   "./data/Versa bank",
    "National Bank of Canada (NBC)"     :   "./data/NATIONAL BANK OF CANADA_ 2022 Annual Report (1).pdf"
    }

def load_doc(path):
    if path.endswith(".pdf"):
        doc = PyPDFLoader(file_path=path)
    else:
        doc = DirectoryLoader(path=path,glob="**/*.pdf")
    document = doc.load()
    context = "\n\n".join([document[i].page_content for i in range(len(document))])
    return context

bank_txt = load_doc(institute_data_paths[institute])[:300000]
bcar_txt = load_doc("./data/Basel Capital Adequacy Reporting (BCAR) 2023 (2).pdf")[:300000]

docs = {
    f"{institute} Annual Report"                :   bank_txt,
    "Basel Capital Adequacy Reporting (BCAR)"   :   bcar_txt,
    }

def compare_answer(question,docs):
    
    retrival_system_template = """You are a helpful assistant, You need to extract as much text as you can which is relater or relevant to the answer of the user question from the context provided.
Do not try to answer the question, just extract the text relevant to the answer of the user question.
Use the following context (delimited by <ctx></ctx>) for finding out the relevant text:

<ctx>
{context}
</ctx>"""
    
    retrival_system_prompt = SystemMessagePromptTemplate.from_template(template=retrival_system_template)
    messages = [retrival_system_prompt,HumanMessage(content=question)]
    compare_chat_prompt = ChatPromptTemplate.from_messages(messages)
    
    summary = dict()
    for doc_name,doc_txt in tqdm(docs.items()):
        summary[doc_name] = llm(compare_chat_prompt.format_prompt(context=doc_txt).to_messages()).content

    compare_context = "\n\n".join([f"Relevant points from {doc_name}:\n\n{doc_summary}" for doc_name,doc_summary in summary.items()])
    
    compare_system_template = """You are a helpful chatbot who has to answer question of a user from the institute {institute} which comes under the BCAR {institute_type} section.
You will be given relevant points from various documents that will help you answer the user question.
Below is a list of relevant points along with the name of the document from where thoes points are from.
Consider all the documents provided to you and answer the question by choosing all the relevant points to the question.
You might have to compare points from more than one document to answer the question.

{context}"""

    compare_system_prompt = SystemMessagePromptTemplate.from_template(template=compare_system_template)
    messages = [compare_system_prompt,HumanMessage(content=question)]
    compare_chat_prompt = ChatPromptTemplate.from_messages(messages)
    response = llm(compare_chat_prompt.format_prompt(institute=institute,institute_type=institute_type,question=question,context=compare_context).to_messages()).content
    return response

question = f"Based on the fiscal year-end mentioned in {institute}'s Annual Report when should it submit BCAR?"

user_input = st.chat_input("Query")



if user_input:
    session.transcript.append(["user",user_input])

    response = compare_answer(user_input,docs)
    session.transcript.append(["assistant",response])

if len(session.transcript)>0:
    # with st.chat_message("assistant"):
    #     st.write(session.transcript[0])
        # st.dataframe(session.transcript[1])
    for message in session.transcript:
        st.chat_message(message[0]).write(message[1])

