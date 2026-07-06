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
                    "Ban co the hoi ve mon an, quan an, dia diem lang man, hoac cac khu vuc ngoai bo du lieu bang Adaptive RAG."
                ),
                "sources": [],
                "debug": None,
                "mode": "agentic",
            }
        ]
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


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


def render_debug(debug_data: dict | None) -> None:
    if not debug_data:
        return

    with st.expander("Xem flow RAG / agent trace", expanded=False):
        route = debug_data.get("route")
        if route:
            st.markdown("**Adaptive route**")
            st.json(route)
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
        search_mode = debug_data.get("search_mode")
        if search_mode:
            st.markdown(f"**Search mode:** `{search_mode}`")
        scope_check = debug_data.get("scope_check")
        if scope_check:
            st.markdown("**Scope check**")
            st.json(scope_check)
        web_results = debug_data.get("web_results")
        if web_results:
            st.markdown("**Web retrieval debug**")
            st.json(web_results)
        web_error = debug_data.get("web_error")
        if web_error:
            st.markdown(f"**Web error:** `{web_error}`")
        memory_debug = debug_data.get("memory_debug")
        if memory_debug:
            st.markdown("**Memory debug**")
            st.json(memory_debug)
        memory_snapshot = debug_data.get("memory_snapshot")
        if memory_snapshot:
            st.markdown("**Memory snapshot**")
            st.json(memory_snapshot)

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
        mode = "agentic"
        retrieval_strategy = "hybrid"
        generation_mode = "template"
        search_mode = "adaptive"
        top_k = 5
        st.markdown(
            """
            <div class='metric-card'>
                <strong>Cau hinh co dinh</strong><br>
                Retrieval: `hybrid` | Search: `adaptive` | Generation: `template` | Top K: `5`
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='metric-card'>
                <strong>Muc tieu</strong><br>
                Khoa cau hinh de chatbot on dinh hon, giam tra loi lech do thay doi qua nhieu tuy chon.
            </div>
            """,
            unsafe_allow_html=True,
        )
    return {
        "mode": mode,
        "top_k": top_k,
        "retrieval_strategy": retrieval_strategy,
        "generation_mode": generation_mode,
        "search_mode": search_mode,
    }


def main() -> None:
    bootstrap_state()
    render_styles()
    chatbot = get_chatbot()
    controls = sidebar_controls()

    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        response = chatbot.answer(
            prompt,
            top_k=int(controls["top_k"]),
            mode=str(controls["mode"]),
            retrieval_strategy=str(controls["retrieval_strategy"]),
            generation_mode=str(controls["generation_mode"]),
            conversation_messages=st.session_state.messages,
            search_mode=str(controls["search_mode"]),
        )
        assistant_message = {
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "debug": response["debug"],
            "mode": controls["mode"],
        }
        st.session_state.messages.append(assistant_message)
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown(
        """
        <div class="title-wrap">
            <h1>Sai Gon Food Chatbot</h1>
            <p>Adaptive RAG duoc co dinh cau hinh de uu tien tinh on dinh khi demo va bao cao.</p>
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
                    render_debug(message.get("debug"))
            st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("### Cach demo")
        st.markdown(
            "- Chatbot dang dung cau hinh co dinh `hybrid + adaptive + top_k=5`.\n"
            "- Thu hoi ve mon an, kieu quan, dia diem hoac khong gian.\n"
            "- Hoi tiep theo kieu `con quan nao khac?`, `Binh Thanh thi sao?`, `so sanh 2 quan nay`.\n"
            "- Cac cau chao hoi va cau ngoai nghiep vu se duoc chan truoc khi retrieve.\n"
            "- Mo `Xem flow RAG / agent trace` de giai thich tai sao chatbot tra loi nhu vay."
        )
        st.markdown("### Cau hoi goi y")
        st.markdown(
            "- Quan nao lang man cho buoi toi o TP.HCM?\n"
            "- Toi muon an vat lot bung o khuya thi nen di dau?\n"
            "- Goi y quan nuong cho gia dinh.\n"
            "- Tim quan co mon chao o TP.HCM.\n"
            "- Con quan nao khac?\n"
            "- Binh Thanh thi sao?\n"
            "- Quan bun bo o Ha Noi thi sao?"
        )

    prompt = st.chat_input("Thu hoi ve mon an, quan an, dia diem o TP.HCM...")
    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": [], "debug": None}
        )
        st.session_state.pending_prompt = prompt
        st.rerun()


if __name__ == "__main__":
    main()
