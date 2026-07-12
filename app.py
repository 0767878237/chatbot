from __future__ import annotations

import streamlit as st

from rag.chatbot import FoodChatbot
from rag.pipeline import build_retriever
from rag.settings import is_debug_enabled


st.set_page_config(
    page_title="FoodMap",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def get_chatbot() -> FoodChatbot:
    retriever = build_retriever(persist_artifacts=False)
    return FoodChatbot(retriever)


def bootstrap_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Xin chao, minh la chatbot am thuc TP.HCM.\n\n"
                    "Ban co the hoi ve mon an, quan an, khu vuc, muc gia hoac phong cach quan ban dang tim."
                ),
                "sources": [],
                "debug": None,
                "mode": "agentic",
            }
        ]
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def build_session_debug(response: dict) -> dict | None:
    if is_debug_enabled():
        return response.get("debug")

    debug = response.get("debug") or {}
    route = debug.get("route") or {}
    retrieved_chunks = debug.get("retrieved_chunks") or []
    compact_chunks: list[dict[str, str]] = []
    for item in retrieved_chunks:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if title:
            compact_chunks.append({"title": title})

    return {
        "route": {
            "selected": route.get("selected", ""),
            "reason": route.get("reason", ""),
            "local_confidence": route.get("local_confidence", 0.0),
            "used_memory": route.get("used_memory", False),
        },
        "retrieved_chunks": compact_chunks,
        "memory_debug": {
            "used_memory": (debug.get("memory_debug") or {}).get("used_memory", False),
        },
    }


def render_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg-start: #0D0E12;
                --bg-end: #14110F;
                --panel: rgba(26, 29, 36, 0.92);
                --panel-strong: rgba(31, 27, 24, 0.97);
                --panel-soft: rgba(37, 41, 50, 0.88);
                --border: rgba(255, 107, 53, 0.18);
                --border-strong: rgba(255, 107, 53, 0.42);
                --text-main: #F3EEE8;
                --text-soft: #B7AAA0;
                --accent: #FF6B35;
                --accent-2: #E85D04;
                --user-panel: #252932;
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(255, 107, 53, 0.12), transparent 24%),
                    radial-gradient(circle at top right, rgba(232, 93, 4, 0.08), transparent 22%),
                    linear-gradient(180deg, var(--bg-start) 0%, var(--bg-end) 100%);
                color: var(--text-main);
            }
            .block-container {
                max-width: 1040px;
                padding-top: 1.2rem;
                padding-bottom: 2.2rem;
            }
            .chat-shell {
                background: linear-gradient(180deg, var(--panel) 0%, var(--panel-strong) 100%);
                border: 1px solid var(--border);
                border-radius: 28px;
                padding: 1rem 1.2rem 1.1rem 1.2rem;
                box-shadow: 0 30px 80px rgba(0, 0, 0, 0.34);
                backdrop-filter: blur(18px);
            }
            .title-wrap {
                padding: 0.2rem 0 1rem 0;
            }
            .hero-card {
                background:
                    linear-gradient(135deg, rgba(31, 27, 24, 0.96) 0%, rgba(26, 29, 36, 0.94) 100%);
                border: 1px solid var(--border);
                border-radius: 28px;
                padding: 1.35rem 1.4rem;
                box-shadow: 0 22px 70px rgba(0, 0, 0, 0.26);
                margin-bottom: 1rem;
            }
            .title-wrap h1 {
                font-size: 2.2rem;
                color: var(--text-main);
                margin: 0;
                letter-spacing: -0.03em;
            }
            .title-kicker {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                color: var(--accent);
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.18em;
                margin-bottom: 0.85rem;
            }
            .title-meta {
                margin-top: 0.6rem;
                color: var(--text-soft);
                font-size: 0.98rem;
            }
            .debug-card {
                border-radius: 16px;
                padding: 0.9rem 1rem;
                background: var(--panel-soft);
                border: 1px dashed var(--border-strong);
                margin-top: 0.8rem;
                color: var(--text-main);
            }
            [data-testid="stChatMessage"] {
                margin-bottom: 0.5rem;
            }
            [data-testid="stChatMessageContent"] {
                color: var(--text-main);
                background: rgba(26, 29, 36, 0.68);
                border: 1px solid rgba(255, 107, 53, 0.10);
                border-radius: 20px;
                padding: 0.2rem 0.35rem;
            }
            [data-testid="stChatMessageContent"] p,
            [data-testid="stChatMessageContent"] div,
            [data-testid="stChatMessageContent"] span,
            [data-testid="stChatMessageContent"] li,
            [data-testid="stChatMessageContent"] strong {
                color: #F3EEE8 !important;
            }
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
                background: var(--user-panel);
                border: 1px solid rgba(243, 238, 232, 0.14);
            }
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p,
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] div,
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] span {
                color: #FAF7F2 !important;
            }
            [data-testid="stChatMessageAvatarAssistant"] {
                background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
                color: #fff5ef;
                border-radius: 14px;
                box-shadow: 0 10px 24px rgba(232, 93, 4, 0.30);
            }
            [data-testid="stChatMessageAvatarUser"] {
                background: #2A2E38;
                color: var(--text-main);
                border-radius: 14px;
                border: 1px solid rgba(243, 238, 232, 0.12);
            }
            [data-testid="stSidebar"],
            [data-testid="collapsedControl"] {
                display: none;
            }
            [data-testid="stExpander"] {
                border: 1px solid var(--border-strong);
                border-radius: 18px;
                background: rgba(31, 27, 24, 0.90);
            }
            [data-testid="stExpander"] summary,
            [data-testid="stExpander"] p,
            [data-testid="stExpander"] label,
            [data-testid="stExpander"] div {
                color: var(--text-main);
            }
            .stChatInputContainer {
                background: transparent !important;
                padding-top: 0.55rem;
            }
            .stChatInputContainer > div {
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
            }
            div[data-testid="stChatInput"] {
                background: #1A1D24 !important;
                border: 1px solid var(--border-strong) !important;
                border-radius: 18px !important;
                box-shadow: 0 18px 44px rgba(0, 0, 0, 0.30) !important;
                overflow: hidden !important;
            }
            div[data-testid="stChatInput"] > div,
            div[data-testid="stChatInput"] > div > div,
            div[data-testid="stChatInput"] > div > div > div {
                background: #1A1D24 !important;
            }
            div[data-testid="stChatInput"] textarea,
            div[data-testid="stChatInput"] input,
            .stChatInput textarea,
            .stChatInput input {
                color: var(--text-main) !important;
                -webkit-text-fill-color: var(--text-main) !important;
                caret-color: var(--accent) !important;
                font-weight: 500 !important;
                background: #1A1D24 !important;
                border: none !important;
                box-shadow: none !important;
            }
            div[data-testid="stChatInput"] textarea::placeholder,
            div[data-testid="stChatInput"] input::placeholder,
            .stChatInput textarea::placeholder,
            .stChatInput input::placeholder {
                color: var(--text-soft) !important;
                -webkit-text-fill-color: var(--text-soft) !important;
                opacity: 0.9 !important;
            }
            div[data-testid="stChatInput"] textarea:focus,
            div[data-testid="stChatInput"] input:focus,
            .stChatInput textarea:focus,
            .stChatInput input:focus {
                background: #1A1D24 !important;
                color: var(--text-main) !important;
                -webkit-text-fill-color: var(--text-main) !important;
                outline: none !important;
                box-shadow: none !important;
            }
            div[data-testid="stChatInput"] textarea::selection,
            div[data-testid="stChatInput"] input::selection,
            .stChatInput textarea::selection,
            .stChatInput input::selection {
                background: rgba(255, 107, 53, 0.30) !important;
                color: #fff7f2 !important;
            }
            div[data-testid="stChatInput"] button,
            .stChatInput button {
                background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%) !important;
                border: 1px solid rgba(255, 107, 53, 0.45) !important;
                color: #fff5ef !important;
                border-radius: 14px !important;
                box-shadow: 0 10px 24px rgba(232, 93, 4, 0.24) !important;
            }
            div[data-testid="stChatInput"] button:hover,
            .stChatInput button:hover {
                background: linear-gradient(135deg, #ff7c4d 0%, #ff6b35 100%) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


CHATBOT_CONFIG = {
    "mode": "agentic",
    "top_k": 5,
    "retrieval_strategy": "hybrid",
    "generation_mode": "template",
    "search_mode": "adaptive",
}


def main() -> None:
    bootstrap_state()
    render_styles()
    chatbot = get_chatbot()

    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        response = chatbot.answer(
            prompt,
            top_k=int(CHATBOT_CONFIG["top_k"]),
            mode=str(CHATBOT_CONFIG["mode"]),
            retrieval_strategy=str(CHATBOT_CONFIG["retrieval_strategy"]),
            generation_mode=str(CHATBOT_CONFIG["generation_mode"]),
            conversation_messages=st.session_state.messages,
            search_mode=str(CHATBOT_CONFIG["search_mode"]),
        )
        assistant_message = {
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "debug": build_session_debug(response),
            "mode": CHATBOT_CONFIG["mode"],
        }
        st.session_state.messages.append(assistant_message)
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown(
        """
        <div class="title-wrap hero-card">
            <div class="title-kicker">Sai Gon Food Intelligence</div>
            <h1>FoodMap</h1>
            <div class="title-meta">Tra cứu ẩm thực và địa điểm ăn uống.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Nhập câu hỏi về quán ăn, địa điểm ăn uống...")
    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": [], "debug": None}
        )
        st.session_state.pending_prompt = prompt
        st.rerun()


if __name__ == "__main__":
    main()
