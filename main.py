import time
from typing import Set

import streamlit as st

from backend.core import run_llm

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    # sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"- {source}\n"
    return sources_string


def main():
    # TODO add Google recaptcha to verify that no auto-bot is asking questions
    # TODO if chat_history is 5, ask user to refer the website pages since chat limit is reached.

    print("Documentation helper demo")
    st.header("Khata Easy - Helper AI Bot")

    prompt = st.text_input("Question", placeholder="Enter your question here...")

    print(f"prompt: {prompt}, chat history: {len(st.session_state['chat_history'])}")

    time.sleep(10)

    if prompt and prompt != "":
        with st.spinner("Generating response..."):
            generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
            # print(f"generated_response: {generated_response}")
            sources = set(
                [
                    doc.metadata["source"]
                    for doc in generated_response["source_documents"]
                ]
            )

            formatted_response = (
                f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
            )

            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formatted_response)
            st.session_state["chat_history"].append(("human", prompt))
            st.session_state["chat_history"].append(("ai", generated_response['answer']))

    if st.session_state["chat_answers_history"]:
        for gen_response, user_query in zip(
            reversed(st.session_state["user_prompt_history"]),
            reversed(st.session_state["chat_answers_history"]),
        ):
            st.chat_message("assistant").write(user_query)
            st.chat_message("user").write(gen_response)


if __name__ == "__main__":
    main()

