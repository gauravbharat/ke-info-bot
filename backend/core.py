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
import langid


# Join the page content from the Document object list
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def has_konkani_characteristics(text: str) -> bool:
    """Check for Konkani-specific words, patterns, and grammatical features."""
    konkani_indicators = [
        # Common verbs and conjugations
        "आसा", "आसा?", "आसा.", "आसलो", "आसले", "आसली", "आसत", "आसतो", "आसते", "आसती",
        "करता", "करते", "करती", "करतो", "करत", "जाता", "जाते", "जातो", "जात", "देता", "देते", "देतो",
        "घेता", "घेते", "घेतो", "मेळटा", "मेळटे", "मेळटो", "पळता", "पळते", "पळतो", "बोलता", "बोलते", "बोलतो",

        # Personal pronouns
        "म्हजो", "म्हजे", "म्हजी", "म्हजें", "म्हाका", "म्हार", "म्हारो", "म्हारे", "म्हारी",
        "तुजो", "तुजे", "तुजी", "तुजें", "तुका", "तुज", "तूं", "तुकार",
        "आमचो", "आमचे", "आमची", "आमचें", "आमकां", "आमी",
        "तुमचो", "तुमचे", "तुमची", "तुमचें", "तुमकां", "तुमी",
        "ताचो", "ताचे", "ताची", "ताचें", "ताका", "तो", "ती", "ते",
        "तिचो", "तिचे", "तिची", "तिचें", "तिका",
        "आपलो", "आपले", "आपली", "आपलें", "आपण",

        # Question words
        "कशें", "कितें", "खंय", "कोण", "कोणूच", "कोणें", "कितलें", "कित्याक", "कशे", "कशी",
        "केन्ना", "केंव", "कसो", "कसले", "कसली", "कसलो",

        # Common adjectives and descriptors
        "वेगळो", "वेगळे", "वेगळी", "वेगळें", "छतो", "छते", "छती", "बरो", "बरे", "बरी",
        "व्हडो", "व्हडे", "व्हडी", "धाकलो", "धाकले", "धाकली", "नवो", "नवे", "नवी",
        "सगळो", "सगळे", "सगळी", "हें", "ह्या", "ही", "हो",

        # Common nouns (daily use)
        "घर", "काम", "वस्त", "मनीस", "बायल", "दादलो", "भुरगें", "शेंकडो", "पयसो", "वेळ",
        "दीस", "रात", "जेवण", "पान", "उदक", "वाट", "जागो", "गांव", "शार",

        # Prepositions and postpositions
        "आदीं", "उपरांत", "भितर", "भायर", "आगी", "मागी", "वांगडा", "सारकें", "परस", "लागीं",

        # Common phrases and expressions
        "देव बरे करू", "कशें आसा?", "कितें नांव?", "घेवन येत", "करून घेवच", "मेळयला", "पळयत",
        "आयलो", "गेलो", "येत", "वत", "आसताना", "करून",

        # Konkani-specific verb forms
        "मार्ता", "मार्ते", "मार्तो", "धरता", "धरते", "धरतो", "सोडता", "सोडते", "सोडतो",
        "उबता", "उबते", "उबतो", "बसता", "बसते", "बसतो", "उठता", "उठते", "उठतो",

        # Negations
        "ना", "न्हय", "नको", "नाका", "नात", "नासलो", "नासले", "नासली",

        # Time-related
        "आयज", "काल", "फाल्यां", "परदीस", "सकाळ", "दनपार", "संज",

        # Numbers (1-10 in Konkani)
        "एक", "दोन", "तीन", "चार", "पांच", "सा", "सात", "आठ", "नव", "धा",

        # Directions
        "उत्तर", "दक्षिण", "पूर्व", "पश्चिम", "वयर", "खाल", "डावे", "उजवे",

        # Family relations
        "आवय", "बापuy", "भाव", "भयण", "घन", "माय", "पूत", "धूव", "नात", "भुरगीं",

        # Common Konkani verbs (root forms)
        "कर", "जा", "ये", "दे", "घे", "मेळ", "पळ", "बोल", "उर", "बस", "उठ", "वाच", "लिह",

        # Konkani-specific grammatical particles
        "च", "तरी", "ना", "क", "ह", "य", "ल", "त", "लो", "ले", "ली", "लें",
    ]

    text_lower = text.lower().strip()

    # Check for exact word matches
    exact_matches = any(indicator in text_lower for indicator in konkani_indicators)

    # Additional pattern matching for Konkani grammar
    konkani_patterns = [
        # Verb endings patterns
        r'\b\w+ता\b', r'\b\w+ते\b', r'\b\w+तो\b', r'\b\w+ती\b',  # Present tense
        r'\b\w+लो\b', r'\b\w+ले\b', r'\b\w+ली\b', r'\b\w+लें\b',  # Past tense
        # Possessive endings
        r'\b\w+चो\b', r'\b\w+चे\b', r'\b\w+ची\b', r'\b\w+चें\b',
        # Common Konkani word structures
        r'\bम्ह\w+', r'\bतु\w+', r'\bआम\w+', r'\bतुम\w+',  # Pronoun prefixes
    ]

    import re
    pattern_matches = any(re.search(pattern, text_lower) for pattern in konkani_patterns)

    # Count occurrences for better confidence
    word_count = sum(1 for indicator in konkani_indicators if indicator in text_lower)

    # Return True if we have reasonable confidence it's Konkani
    return exact_matches or pattern_matches or word_count >= 2


def detect_user_language(text: str) -> str:
    """Detects the language of the input text and returns a KhataEasy code."""
    lang_map = {
        "gom": "gom",  # Konkani
        "en": "en",  # English
        "gu": "gu",  # Gujarati
        "hi": "hi",  # Hindi
        "mr": "mr",  # Marathi
        "other": "en",  # Default to English if detection is uncertain
    }

    try:
        detected_lang, confidence_score = langid.classify(text)

        # langid returns negative log-likelihood - LOWER values are better
        # Typical good confidence: -50 to -150 (much lower than -1)
        # Poor confidence: closer to 0 or positive values
        print(f"langid detected: {detected_lang}, confidence (negative log-likelihood): {confidence_score}")

        # Convert negative log-likelihood to a more intuitive "certainty" score
        # Lower negative values = more certain, higher negative values = less certain
        certainty = min(1.0, max(0.0, (-confidence_score) / 100))  # Rough conversion
        print(f"Converted certainty: {certainty:.2f}")

        detected_iso_code = detected_lang

        if detected_lang == "hi" and confidence_score > -150 and has_konkani_characteristics(text):
            print("Overriding Hindi detection to Konkani based on characteristics")
            detected_iso_code = "gom"

        # Check against your known list of languages
        if detected_iso_code in lang_map:
            return lang_map[detected_iso_code]
        elif detected_iso_code in ["bn", "pa", "ta", "te"]:  # Common Indian languages not on your list
            # Fallback to English or a specific handler if needed
            return "en"
        else:
            return "en"  # Default fallback

    except Exception as e:
        print(f"Language detection failed: {e}")
        return "en"  # Default to English on failure


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    """Invoke LLM for the passed string and chat history"""
    print("*** INVOKE LLM ***")
    print(f"query: {query}")
    print(f"chat_history total: {len(chat_history)}")

    print("\n* STEP 0: DETECT LANG *")
    user_lang_code = detect_user_language(query)
    print(f"Detected KhataEasy Language Code: {user_lang_code}")

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
    # folder_name = f"ke_info_vector_store_{user_lang_code}"
    folder_path = os.path.join("backend", f"ke_info_vector_store_{user_lang_code}")
    vector_store = FAISS.load_local(
        folder_path, embeddings=embeddings, allow_dangerous_deserialization=True
    )

    # 2. init llm
    print("* STEP 2 *")
    print("Initialising GoogleGenerativeAI model gemini-2.5-flash-lite...")
    # TODO try changing model for answer accuracy from vector store
    llm = GoogleGenerativeAI(
        api_key=st.secrets.get("GOOGLE_API_KEY"), model="gemini-2.5-flash-lite"
    )

    # 3. create retrieval chat prompt and stuff documents
    print("* STEP 3 *")
    print("get the retrieval qa chat prompt...")
    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    retrieval_qa_chat_prompt_template = """
    SYSTEM
    You are an expert multilingual assistant for KhataEasy accounting software. Your operation is governed by **STRICT, 
    MANDATORY** rules.
    
    **MANDATORY RULES:**
    
    1.  **Language Lock (Output):** The **entire response MUST be in the same language as the user's current question**. 
        **NEVER** mix languages or use a different language unless explicitly asked to translate.
    
    2.  **Context Trust:** All provided context is already filtered to be in the correct language. Answer **ONLY** 
    using the information within the `<context>` block. Do not question the language of the context.
    
    3.  **Formatting Lock (Suppression):** **NEVER** output the text `<context>`, `</context>`, `SYSTEM`, `HUMAN`, 
        or `MESSAGES_LIST` in your final response. These are internal instructions only.
    
    4.  **Answering Protocol:** 
        * **Pricing/Cost:** If the query is about cost, price, or subscription, prioritize context related to **trial 
        and regular paid options**. Be specific about pricing details when available.
        * **Chat History:** If the exact answer is present in the `chat_history`, politely remind the user this was 
        already answered.
        * **Clarity:** Provide clear, concise answers. If multiple options exist (like different subscription plans), 
        list them clearly.
    
    5.  **Failure Protocol (Final Rule):** If a relevant answer is **not found** in the context, you **MUST** respond 
        with the following fallback message in the user's current language: 
        *"For more information, please visit https://khataeasy.com"*
    
    <context>
    {context}
    </context>
    
    MESSAGES_LIST
    {chat_history}
    
    HUMAN
    {input}
    """

    retrieval_qa_chat_prompt = PromptTemplate.from_template(
        retrieval_qa_chat_prompt_template
    )

    print("create a chain that stuffs documents into retrieval chat prompt...")
    stuff_documents_chain = create_stuff_documents_chain(
        llm=llm, prompt=retrieval_qa_chat_prompt
    )

    # 4. get rephrase prompt for chat history and create chat history aware retriever
    print("* STEP 4 *")
    print("get the rephrase prompt...")
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    print(
        "create history aware retriever with the llm, vector store as retriever and rephrase prompt..."
    )
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=vector_store.as_retriever(), prompt=rephrase_prompt
    )

    # 5. create retrieval chain
    print("* STEP 5 *")
    print(
        "create retrieval chain passing the retriever as the history aware retrieve and stuffed docs chain..."
    )
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


if __name__ == "__main__":
    res = run_llm(query="subscription cost?", chat_history=[])
    # print(f"res: {res}")
    # gom: अॅप कसो वेगळो आसा
