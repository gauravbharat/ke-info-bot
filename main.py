import random
import time
from typing import Dict, List, Set, Tuple

import streamlit as st

from backend.core import run_llm

MAX_CHAT_HISTORY = 5
CAPTCHA_TIMEOUT_SECONDS = 300
CAPTCHA_MAX_ATTEMPTS = 3
CAPTCHA_INPUT_RANGE = (-100, 100)

SESSION_DEFAULTS: Dict[str, object] = {
    "user_prompt_history": [],
    "chat_answers_history": [],
    "chat_history": [],
    "captcha_verified": False,
    "captcha_attempts": 0,
    "captcha_generated_time": 0.0,
    "welcome_toast_shown": False,
}

SUGGESTIONS = {
    ":blue[:material/local_library:] What is Khata Easy?": (
        "What is Khata Easy, what is it great at, and what can I do with it?"
    ),
    ":green[:material/shield_lock:] મારો ડેટા કેવી રીતે સંગ્રહિત અને સુરક્ષિત છે?": (
        "મારો ડેટા કેવી રીતે સંગ્રહિત અને સુરક્ષિત છે?"
    ),
    ":orange[:material/help:] What is the subscription cost?": (
        "what is the subscription cost?"
    ),
    ":violet[:material/auto_awesome:] 会计数据是如何存储的？": (
        "会计数据是如何存储的？"
    ),
    ":red[:material/rocket_launch:] मैं ऐप का उपयोग कैसे प्रारंभ करूँ?": (
        "ऐप का उपयोग"
    ),
}


def initialize_session_state() -> None:
    """Initialize all required session keys once."""
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            if isinstance(default, list):
                st.session_state[key] = list(default)
            else:
                st.session_state[key] = default


@st.dialog("Legal disclaimer")
def show_disclaimer_dialog() -> None:
    st.caption(
        """
            This AI chatbot is powered by Streamlit, Google Gemini and public Khata Easy
            information. Answers may be inaccurate, inefficient, or biased.
            Any use or decisions based on such answers should include reasonable
            practices including human oversight to ensure they are safe,
            accurate, and suitable for your intended purpose. Khata Easy is not
            liable for any actions, losses, or damages resulting from the use
            of the chatbot. Do not enter any private, sensitive, personal, or
            regulated data. By using this chatbot, you acknowledge and agree
            that input you provide and answers you receive (collectively,
            “Content”) may be used by Streamlit to provide, maintain, develop,
            and improve their respective offerings. For more
            information on how Streamlit may use your Content, see
            https://streamlit.io/terms-of-service.
        """
    )


def generate_math_captcha() -> Tuple[str, int]:
    """Generate a simple math CAPTCHA"""
    # Generate random numbers for the math problem
    num1 = random.randint(1, 15)
    num2 = random.randint(1, 15)
    operation = random.choice(["+", "-", "*"])

    if operation == "+":
        answer = num1 + num2
        problem = f"{num1} + {num2}"
    elif operation == "-":
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


def simple_math_captcha() -> bool:
    """Display and verify math CAPTCHA"""
    if not st.session_state["captcha_verified"]:
        # Generate new CAPTCHA if needed or after 5 minutes
        current_time = time.time()
        if (
            "captcha_problem" not in st.session_state
            or "captcha_answer" not in st.session_state
            or current_time - st.session_state["captcha_generated_time"]
            > CAPTCHA_TIMEOUT_SECONDS
        ):

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
                min_value=CAPTCHA_INPUT_RANGE[0],
                max_value=CAPTCHA_INPUT_RANGE[1],
                step=1,
                key="captcha_user_answer",
            )

            # Verify button
            if st.button(
                "✅ Verify I'm Human", type="primary", use_container_width=True
            ):
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

                    if st.session_state["captcha_attempts"] >= CAPTCHA_MAX_ATTEMPTS:
                        st.warning(
                            "🚫 Multiple failed attempts. Please refresh the page."
                        )
                    else:
                        st.rerun()

        st.info(
            "💡 This helps us prevent automated bots and ensure quality service for all users."
        )
        return False

    return True


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = sorted(source_urls)
    sources_string = "sources:\n"
    for source in sources_list:
        sources_string += f"- {source.replace('?refresh=fullcrawl', '')}\n"
    return sources_string


def get_theme_mode() -> str:
    """Get current theme mode (light/dark)"""
    try:
        # Method 1: Check Streamlit's theme config
        theme = st.get_option("theme.base")
        return "dark" if theme == "dark" else "light"
    except Exception:
        # Method 2: Fallback to checking background color
        try:
            background_color = st.get_option("theme.backgroundColor")
            return "dark" if background_color == "#0E1117" else "light"
        except Exception:
            return "light"  # Default to light


def render_conversation_history(
    title: str,
    user_prompts: List[str],
    answers: List[str],
    *,
    compact: bool = False,
) -> None:
    """Render conversation history in reverse chronological order."""
    if not answers:
        return

    st.markdown("---")
    st.subheader(title)
    for user_query, gen_response in zip(reversed(user_prompts), reversed(answers)):
        if compact:
            with st.expander(
                f"Q: {user_query[:60]}..." if len(user_query) > 60 else f"Q: {user_query}"
            ):
                st.chat_message("user").write(user_query)
                st.chat_message("assistant").write(gen_response)
            continue

        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(gen_response)
        st.markdown("---")


def render_sidebar_status(current_usage: int) -> None:
    """Show usage and CAPTCHA status."""
    st.sidebar.markdown("### 📊 Usage Status")
    st.sidebar.write(
        f"**Questions asked:** {current_usage}/{MAX_CHAT_HISTORY}"
    )

    if current_usage >= MAX_CHAT_HISTORY:
        st.sidebar.error("🚨 Rate limit reached")
    elif current_usage >= MAX_CHAT_HISTORY - 1:
        st.sidebar.warning("⚠️ Last question remaining")
    elif current_usage > 0:
        st.sidebar.success("✅ Questions available")

    if st.session_state["captcha_verified"]:
        st.sidebar.success("🤖 CAPTCHA: Verified")
        verification_time = st.session_state.get("captcha_verified_time", 0)
        if verification_time:
            elapsed = int((time.time() - verification_time) / 60)
            st.sidebar.write(f"Verified: {elapsed} min ago")
    else:
        st.sidebar.warning("🤖 CAPTCHA: Pending")


def handle_prompt_submission(prompt: str) -> None:
    """Run the assistant, persist state, and render history."""
    with st.spinner("🔍 Searching for the best answer..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        sources = {
            doc.metadata["source"] for doc in generated_response["source_documents"]
        }
        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["answer"]))

    render_conversation_history(
        "📝 Conversation History",
        st.session_state["user_prompt_history"],
        st.session_state["chat_answers_history"],
    )


def main() -> None:
    initialize_session_state()
    st.set_page_config(
        page_title="Khata Easy - AI Assistant",
        page_icon="assets/favicon.ico",
        layout="centered",
    )

    # Add meta tags for SEO
    st.markdown(
        """
        <meta name="description" content="Khata Easy AI Assistant - Get instant answers about our secure accounting software, pricing, features, and how to get started. Available in multiple languages.">
        <meta name="keywords" content="khata easy, accounting software, small business accounting, gst billing, inventory management, secure accounting, multi-language support">
        <meta name="author" content="Khata Easy">
        <meta property="og:title" content="Khata Easy - AI Assistant">
        <meta property="og:description" content="Get instant answers about Khata Easy accounting software features, pricing, and security.">
        <meta property="og:type" content="website">
    """,
        unsafe_allow_html=True,
    )

    image_path = (
        "assets/kp_logo_light.png"
        if get_theme_mode() == "light"
        else "assets/kp_logo.png"
    )
    st.image(image_path, width=150)
    st.header("Khata Easy - Helper AI Bot")

    # Rate limit configuration
    current_usage = len(st.session_state["user_prompt_history"])
    render_sidebar_status(current_usage)

    # Check rate limit
    if current_usage >= MAX_CHAT_HISTORY:
        st.error(
            f"""
        🚨 **Rate Limit Reached**

        You've reached the maximum number of questions ({MAX_CHAT_HISTORY}) allowed in this session. 

        **For more information:**
        - 📚 Browse our documentation at [https://khataeasy.com](https://khataeasy.com)

        Thank you for understanding!
        """
        )
        render_conversation_history(
            "📝 Your Conversation History",
            st.session_state["user_prompt_history"],
            st.session_state["chat_answers_history"],
            compact=True,
        )

        return

    # CAPTCHA Verification Section
    captcha_passed = simple_math_captcha()

    if not captcha_passed:
        return

    # Main chat interface (only shown after CAPTCHA verification).
    if current_usage < 1 and not st.session_state["welcome_toast_shown"]:
        st.toast("✅ Verification complete! You can now ask your questions.", icon="✅")
        st.session_state["welcome_toast_shown"] = True

    prompt = st.text_input(
        "💬 Ask your question:",
        placeholder="Type your question about Khata Easy here...",
        key="question_input",
    )

    if not prompt:
        prompt = st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion",
        )

    st.button(
        "&nbsp;:small[:gray[:material/balance: Legal disclaimer]]",
        type="tertiary",
        on_click=show_disclaimer_dialog,
    )

    if prompt:
        handle_prompt_submission(prompt)


if __name__ == "__main__":
    main()
