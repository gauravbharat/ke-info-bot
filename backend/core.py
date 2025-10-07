import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain import hub
from typing import List, Dict, Any
import streamlit as st


# Join the page content from the Document object list
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    """Invoke LLM for the passed string and chat history"""
    print("*** INVOKE LLM ***")
    print(f"query: {query}")
    print(f"chat_history total: {len(chat_history)}")
    # 1. Load local vector store
    print("\n* STEP 1 *")
    print("Loading embeddings...")
    embeddings = OpenAIEmbeddings(
        api_key=st.secrets.get("OPENAI_API_KEY"),
        model="text-embedding-3-small",
        show_progress_bar=False,
        chunk_size=50,
        retry_min_seconds=10,
    )

    print("Loading vector store ke_info_vector_store...")
    folder_path = os.path.join("backend", "ke_info_vector_store")
    vector_store = FAISS.load_local(
        folder_path, embeddings=embeddings, allow_dangerous_deserialization=True
    )

    # 2. init llm
    print("* STEP 2 *")
    print("Initialising GoogleGenerativeAI model gemini-2.5-flash-lite...")
    llm = GoogleGenerativeAI(api_key=st.secrets.get("GOOGLE_API_KEY"), model="gemini-2.5-flash-lite")

    # 3. create retrieval chat prompt and stuff documents
    print("* STEP 3 *")
    print("get the retrieval qa chat prompt...")
    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    retrieval_qa_chat_prompt_template = '''
      SYSTEM
      Answer any use questions based solely on the context below and use translation based on the question or input 
      language, use en or English by default if the input language is not known:

      <context>
      {context}
      </context>

      MESSAGES_LIST
      chat_history
      
      if anyone asks for cost, price or subscription cost, try to search subscription options of trial and regular paid 
      instead of commission in the context. if the answer is already provided in the chat_history, say so that already 
      answered. if the answer is not provided in the context say "Sorry, answer is outside of my context data. For more 
      information, please visit https://khataeasy.com"  
      HUMAN
      {input}
    '''

    retrieval_qa_chat_prompt = PromptTemplate.from_template(retrieval_qa_chat_prompt_template)

    print("create a chain that stuffs documents into retrieval chat prompt...")
    stuff_documents_chain = create_stuff_documents_chain(
        llm=llm, prompt=retrieval_qa_chat_prompt
    )

    # 4. get rephrase prompt for chat history and create chat history aware retriever
    print("* STEP 4 *")
    print("get the rephrase prompt...")
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    print("create history aware retriever with the llm, vector store as retriever and rephrase prompt...")
    history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=vector_store.as_retriever(),
                                                             prompt=rephrase_prompt)

    # 5. create retrieval chain
    print("* STEP 5 *")
    print("create retrieval chain passing the retriever as the history aware retrieve and stuffed docs chain...")
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    # 6. Invoke retrieval chain
    print("* STEP 6 *")
    print("invoke the retrieval chain passing the query and chat history...")
    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    # print('answer', result["answer"])
    # print('input', result["input"])
    # print('context', result["context"])
    new_result = {
        "query": result["input"],
        "answer": result["answer"],
        "source_documents": result["context"],
    }
    print("* RESULT SUCCESS *")
    # print(new_result["answer"])
    # print()

    return new_result


if __name__ == '__main__':
    res = run_llm(query="subscription cost?", chat_history=[])
    # print(f"res: {res}")
    # gom: अॅप कसो वेगळो आसा

