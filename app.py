from __future__ import annotations

import streamlit as st

from rag.chatbot import FoodChatbot
from rag.pipeline import build_retriever


st.set_page_config(
    page_title="Sai Gon Food Chatbot",
    page_icon="🍜",
    layout="wide",
)


@st.cache_resource
def get_chatbot() -> FoodChatbot:
    retriever = build_retriever()
    return FoodChatbot(retriever)


def bootstrap_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Xin chào, mình là chatbot ẩm thực TP.HCM.\n\n"
                    "Bạn có thể hỏi về quán ăn tối, món ăn vặt, địa điểm lãng mạn "
                    "hoặc các gợi ý trong bộ dữ liệu hiện tại."
                ),
                "sources": [],
                "debug": None,
            }
        ]


def render_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(255, 229, 180, 0.45), transparent 30%),
                    linear-gradient(180deg, #fff8ef 0%, #f5efe6 100%);
            }
            .block-container {
                max-width: 1080px;
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            .chat-shell {
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(126, 76, 39, 0.15);
                border-radius: 28px;
                padding: 1rem 1.2rem;
                box-shadow: 0 24px 80px rgba(91, 58, 34, 0.10);
                backdrop-filter: blur(14px);
            }
            .title-wrap {
                padding: 0.2rem 0 1rem 0;
            }
            .title-wrap h1 {
                font-size: 2rem;
                color: #5c3417;
                margin-bottom: 0.2rem;
            }
            .title-wrap p {
                color: #7d5b46;
                margin: 0;
            }
            .source-card {
                border-radius: 16px;
                padding: 0.8rem 1rem;
                margin-top: 0.6rem;
                background: #fff6ea;
                border: 1px solid #f0dbc0;
            }
            .debug-card {
                border-radius: 16px;
                padding: 0.9rem 1rem;
                background: #f4f1ec;
                border: 1px dashed #ceb79a;
                margin-top: 0.8rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list) -> None:
    if not sources:
        return

    st.markdown("**Nguồn tham khảo**")
    for item in sources:
        document = item.chunk.document
        addresses = "; ".join(document.addresses) if document.addresses else "Không có địa chỉ"
        st.markdown(
            (
                "<div class='source-card'>"
                f"<strong>{document.title}</strong><br>"
                f"Chunk: {item.chunk.chunk_id}<br>"
                f"Địa chỉ: {addresses}<br>"
                f"Điểm liên quan: {item.score:.3f}<br><br>"
                f"{item.chunk.text}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_debug(debug_data: dict | None) -> None:
    if not debug_data:
        return

    with st.expander("Xem flow RAG", expanded=False):
        retrieved_chunks = debug_data.get("retrieved_chunks", [])
        if retrieved_chunks:
            st.markdown("**Các chunk được retrieve**")
            for chunk in retrieved_chunks:
                st.markdown(
                    (
                        "<div class='debug-card'>"
                        f"<strong>{chunk['title']}</strong><br>"
                        f"Score: {chunk['score']}<br>"
                        f"Từ khớp: {', '.join(chunk['matched_terms']) or 'Không có'}<br><br>"
                        f"{chunk['content']}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        st.markdown("**Prompt preview**")
        st.code(debug_data.get("prompt_preview", ""), language="text")


def main() -> None:
    bootstrap_state()
    render_styles()
    chatbot = get_chatbot()

    st.markdown(
        """
        <div class="title-wrap">
            <h1>Sai Gon Food Chatbot</h1>
            <p>Ban chat hoat dong theo retrieval + template answer de ban de hoc flow RAG tren may CPU.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                render_sources(message.get("sources", []))
                render_debug(message.get("debug"))
        st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Thu hoi ve mon an, quan an, dia diem o TP.HCM...")
    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": [], "debug": None}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        response = chatbot.answer(prompt)
        assistant_message = {
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "debug": response["debug"],
        }
        st.session_state.messages.append(assistant_message)
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
            render_sources(response["sources"])
            render_debug(response["debug"])


if __name__ == "__main__":
    main()
