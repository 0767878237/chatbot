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
                    "Xin chao, minh la chatbot am thuc TP.HCM.\n\n"
                    "Ban co the hoi ve mon an, quan an, dia diem lang man hoac so sanh giua che do Baseline RAG va Agentic RAG."
                ),
                "sources": [],
                "debug": None,
                "mode": "agentic",
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
                max-width: 1120px;
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            .chat-shell {
                background: rgba(255, 255, 255, 0.8);
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
            .metric-card {
                border-radius: 18px;
                padding: 0.9rem 1rem;
                background: rgba(255, 255, 255, 0.7);
                border: 1px solid rgba(126, 76, 39, 0.12);
                margin-bottom: 0.75rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list) -> None:
    if not sources:
        return

    st.markdown("**Nguon tham khao**")
    for item in sources:
        document = item.chunk.document
        addresses = "; ".join(document.addresses) if document.addresses else "Khong co dia chi"
        st.markdown(
            (
                "<div class='source-card'>"
                f"<strong>{document.title}</strong><br>"
                f"Chunk: {item.chunk.chunk_id}<br>"
                f"Dia chi: {addresses}<br>"
                f"Diem lien quan: {item.score:.3f}<br><br>"
                f"{item.chunk.text}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_debug(debug_data: dict | None) -> None:
    if not debug_data:
        return

    with st.expander("Xem flow RAG / agent trace", expanded=False):
        strategy = debug_data.get("retrieval_strategy")
        if strategy:
            st.markdown(f"**Retrieval strategy:** `{strategy}`")
        generation_mode = debug_data.get("generation_mode")
        if generation_mode:
            st.markdown(f"**Generation mode:** `{generation_mode}`")
        generation_debug = debug_data.get("generation_debug")
        if generation_debug:
            st.markdown("**Generation debug**")
            st.json(generation_debug)

        query_analysis = debug_data.get("query_analysis")
        if query_analysis:
            st.markdown("**Phan tich truy van**")
            st.json(query_analysis)

        agent_steps = debug_data.get("agent_steps", [])
        if agent_steps:
            st.markdown("**Cac buoc agent da thuc hien**")
            for step in agent_steps:
                st.markdown(
                    (
                        "<div class='debug-card'>"
                        f"<strong>{step['name']}</strong><br>"
                        f"{step['detail']}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                st.json(step.get("payload", {}))

        retrieved_chunks = debug_data.get("retrieved_chunks", [])
        if retrieved_chunks:
            st.markdown("**Cac chunk duoc retrieve**")
            for chunk in retrieved_chunks:
                st.markdown(
                    (
                        "<div class='debug-card'>"
                        f"<strong>{chunk['title']}</strong><br>"
                        f"Score: {chunk['score']}<br>"
                        f"Score breakdown: {chunk.get('score_breakdown', {})}<br>"
                        f"Tu khop: {', '.join(chunk['matched_terms']) or 'Khong co'}<br><br>"
                        f"{chunk['content']}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        st.markdown("**Prompt preview**")
        st.code(debug_data.get("prompt_preview", ""), language="text")


def sidebar_controls() -> dict[str, object]:
    with st.sidebar:
        st.markdown("## Che do demo")
        mode = st.radio(
            "Chon che do chat",
            options=["agentic", "baseline"],
            format_func=lambda value: "Agentic RAG" if value == "agentic" else "Baseline RAG",
            index=0,
        )
        retrieval_strategy = st.selectbox(
            "Chien luoc retrieval",
            options=["hybrid", "lexical", "semantic"],
            index=0,
            help="Hybrid la mac dinh de deploy mien phi: ket hop TF-IDF va latent semantic scoring tu scikit-learn.",
        )
        generation_mode = st.selectbox(
            "Che do generation",
            options=["template", "ollama"],
            index=0,
            help="Template dung cho deploy cloud. Ollama dung khi demo local va may dang chay model local.",
        )
        top_k = st.slider("So chunk retrieve", min_value=2, max_value=6, value=4)
        st.markdown(
            """
            <div class='metric-card'>
                <strong>Deploy note</strong><br>
                Khi deploy Streamlit Cloud hay de `generation = template`. Khi demo local, ban co the bat `ollama`.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='metric-card'>
                <strong>Gia tri do an</strong><br>
                Agentic RAG hien phan tich truy van, nhieu lan retrieve, loc va rerank thay vi chi tim 1 lan.
            </div>
            """,
            unsafe_allow_html=True,
        )
    return {
        "mode": mode,
        "top_k": top_k,
        "retrieval_strategy": retrieval_strategy,
        "generation_mode": generation_mode,
    }


def main() -> None:
    bootstrap_state()
    render_styles()
    chatbot = get_chatbot()
    controls = sidebar_controls()

    st.markdown(
        """
        <div class="title-wrap">
            <h1>Sai Gon Food Chatbot</h1>
            <p>Ban demo do an voi hai che do: Baseline RAG de lam moc so sanh va Agentic RAG de the hien truy van nhieu buoc.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([2.2, 1])
    with left_col:
        with st.container():
            st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    render_sources(message.get("sources", []))
                    render_debug(message.get("debug"))
            st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("### Cach demo")
        st.markdown(
            "- Thu hoi ve mon an, kieu quan, dia diem hoac khong gian.\n"
            "- Chuyen qua lai giua `Baseline RAG` va `Agentic RAG`.\n"
            "- Mo `Xem flow RAG / agent trace` de giai thich tai sao chatbot tra loi nhu vay."
        )
        st.markdown("### Cau hoi goi y")
        st.markdown(
            "- Quan nao lang man cho buoi toi o TP.HCM?\n"
            "- Toi muon an vat lot bung o khuya thi nen di dau?\n"
            "- Goi y quan nuong cho gia dinh.\n"
            "- Tim quan co mon chao o TP.HCM."
        )

    prompt = st.chat_input("Thu hoi ve mon an, quan an, dia diem o TP.HCM...")
    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": [], "debug": None}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        response = chatbot.answer(
            prompt,
            top_k=int(controls["top_k"]),
            mode=str(controls["mode"]),
            retrieval_strategy=str(controls["retrieval_strategy"]),
            generation_mode=str(controls["generation_mode"]),
        )
        assistant_message = {
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "debug": response["debug"],
            "mode": controls["mode"],
        }
        st.session_state.messages.append(assistant_message)
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
            render_sources(response["sources"])
            render_debug(response["debug"])


if __name__ == "__main__":
    main()
