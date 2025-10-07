from typing import Set
import streamlit as st
import random
import time
from backend.core import run_llm

# Initialize session state variables
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "captcha_verified" not in st.session_state:
    st.session_state["captcha_verified"] = False

if "captcha_attempts" not in st.session_state:
    st.session_state["captcha_attempts"] = 0

if "captcha_generated_time" not in st.session_state:
    st.session_state["captcha_generated_time"] = 0

MAX_CHAT_HISTORY = 5


def generate_math_captcha():
    """Generate a simple math CAPTCHA"""
    # Generate random numbers for the math problem
    num1 = random.randint(1, 15)
    num2 = random.randint(1, 15)
    operation = random.choice(['+', '-', '*'])

    if operation == '+':
        answer = num1 + num2
        problem = f"{num1} + {num2}"
    elif operation == '-':
        # Ensure positive result
        num1, num2 = max(num1, num2), min(num1, num2)
        answer = num1 - num2
        problem = f"{num1} - {num2}"
    else:  # multiplication
        num1 = random.randint(1, 10)
        num2 = random.randint(1, 5)
        answer = num1 * num2
        problem = f"{num1} × {num2}"

    return problem, answer


def simple_math_captcha():
    """Display and verify math CAPTCHA"""
    if not st.session_state["captcha_verified"]:
        # Generate new CAPTCHA if needed or after 5 minutes
        current_time = time.time()
        if ("captcha_problem" not in st.session_state or
                "captcha_answer" not in st.session_state or
                current_time - st.session_state["captcha_generated_time"] > 300):  # 5 minutes

            problem, answer = generate_math_captcha()
            st.session_state["captcha_problem"] = problem
            st.session_state["captcha_answer"] = answer
            st.session_state["captcha_generated_time"] = current_time

        st.markdown("---")
        st.markdown("### 🤖 Human Verification Required")
        st.write("Please solve this simple math problem to prevent automated queries:")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Display CAPTCHA problem
            st.markdown(f"### 🧮 {st.session_state.captcha_problem} = ?")

            # Answer input
            answer = st.number_input(
                "Enter your answer:",
                min_value=-100,
                max_value=100,
                step=1,
                key="captcha_user_answer"
            )

            # Verify button
            if st.button("✅ Verify I'm Human", type="primary", use_container_width=True):
                if answer == st.session_state["captcha_answer"]:
                    st.session_state["captcha_verified"] = True
                    st.session_state["captcha_verified_time"] = time.time()
                    st.session_state["captcha_attempts"] = 0
                    st.success("✅ Verification successful!")
                    st.balloons()
                    st.rerun()
                else:
                    st.session_state["captcha_attempts"] += 1
                    st.error("❌ Incorrect answer. Please try again.")

                    # Generate new CAPTCHA after failed attempt
                    problem, answer = generate_math_captcha()
                    st.session_state["captcha_problem"] = problem
                    st.session_state["captcha_answer"] = answer
                    st.session_state["captcha_generated_time"] = time.time()

                    if st.session_state["captcha_attempts"] >= 3:
                        st.warning("🚫 Multiple failed attempts. Please refresh the page.")
                    else:
                        st.rerun()

        st.info("💡 This helps us prevent automated bots and ensure quality service for all users.")
        return False

    return True


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"- {source}\n"
    return sources_string


def main():
    global MAX_CHAT_HISTORY
    # Set page config
    st.set_page_config(
        page_title="Khata Easy - AI Assistant",
        page_icon="🤖",
        layout="centered"
    )

    print("Documentation helper demo")
    st.header("Khata Easy - Helper AI Bot")

    # Rate limit configuration
    current_usage = len(st.session_state["user_prompt_history"])

    # Show usage status in sidebar
    st.sidebar.markdown("### 📊 Usage Status")
    st.sidebar.write(f"**Questions asked:** {len(st.session_state['user_prompt_history'])}/{MAX_CHAT_HISTORY}")

    if current_usage >= MAX_CHAT_HISTORY:
        st.sidebar.error("🚨 Rate limit reached")
    elif current_usage >= MAX_CHAT_HISTORY - 1:
        st.sidebar.warning("⚠️ Last question remaining")
    elif current_usage > 0:
        st.sidebar.success("✅ Questions available")

    # Show CAPTCHA status
    if st.session_state["captcha_verified"]:
        st.sidebar.success("🤖 CAPTCHA: Verified")
        verification_time = st.session_state.get("captcha_verified_time", 0)
        if verification_time:
            elapsed = int((time.time() - verification_time) / 60)
            st.sidebar.write(f"Verified: {elapsed} min ago")
    else:
        st.sidebar.warning("🤖 CAPTCHA: Pending")

    # Check rate limit
    if current_usage >= MAX_CHAT_HISTORY:
        st.error(f"""
        🚨 **Rate Limit Reached**

        You've reached the maximum number of questions ({MAX_CHAT_HISTORY}) allowed in this session. 

        **For more information:**
        - 📚 Browse our documentation at [https://khataeasy.com](https://khataeasy.com)

        Thank you for understanding!
        """)
        #         - 🔄 Refresh the page to start a new session

        # Show conversation history for reference
        if st.session_state["chat_answers_history"]:
            st.markdown("---")
            st.subheader("📝 Your Conversation History")
            for user_query, gen_response in zip(
                    reversed(st.session_state["user_prompt_history"]),
                    reversed(st.session_state["chat_answers_history"])
            ):
                with st.expander(f"Q: {user_query[:60]}..." if len(user_query) > 60 else f"Q: {user_query}"):
                    st.chat_message("user").write(user_query)
                    st.chat_message("assistant").write(gen_response)

        return

    # CAPTCHA Verification Section
    captcha_passed = simple_math_captcha()

    if not captcha_passed:
        return

    # Main Chat Interface (only shown after CAPTCHA verification)
    if current_usage < 1:
        st.success("✅ Verification complete! You can now ask your questions.")

    remaining = (MAX_CHAT_HISTORY - current_usage) - 1
    if remaining > 0:
        st.info(f"💡 You have **{remaining}** question{'s' if remaining > 1 else ''} remaining in this session")

    prompt = st.text_input(
        "💬 Ask your question:",
        placeholder="Type your question about Khata Easy here...",
        key="question_input"
    )

    print(f"prompt: {prompt}, chat history: {len(st.session_state['chat_history'])}")

    if prompt and prompt != "":
        with st.spinner("🔍 Searching for the best answer..."):
            generated_response = run_llm(
                query=prompt,
                chat_history=st.session_state["chat_history"]
            )

            sources = set([
                doc.metadata["source"]
                for doc in generated_response["source_documents"]
            ])

            formatted_response = (
                f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
            )

            # Update session state
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formatted_response)
            st.session_state["chat_history"].append(("human", prompt))
            st.session_state["chat_history"].append(("ai", generated_response['answer']))

            # Display conversation history
            if st.session_state["chat_answers_history"]:
                st.markdown("---")
                st.subheader("📝 Conversation History")

                for user_query, gen_response in zip(
                        reversed(st.session_state["user_prompt_history"]),
                        reversed(st.session_state["chat_answers_history"])
                ):
                    st.chat_message("user").write(user_query)
                    st.chat_message("assistant").write(gen_response)
                    st.markdown("---")

    # # Admin section in sidebar
    # with st.sidebar:
    #     st.markdown("---")
    #     st.markdown("### 🔧 Admin")
    #
    #     if st.button("🔄 Reset Session", help="Clear all chat history and start fresh"):
    #         for key in list(st.session_state.keys()):
    #             del st.session_state[key]
    #         st.rerun()
    #
    #     # Show CAPTCHA info for debugging
    #     if st.checkbox("Show debug info", False):
    #         if "captcha_problem" in st.session_state:
    #             st.write(f"CAPTCHA: {st.session_state.captcha_problem} = {st.session_state.captcha_answer}")
    #         st.write("Session keys:", list(st.session_state.keys()))


if __name__ == "__main__":
    main()

