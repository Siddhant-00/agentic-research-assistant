import uuid
import streamlit as st

from graph.workflow import app_graph


st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🤖"
)

st.title(
    "🤖 Agentic Research Assistant"
)

st.markdown(
    """
Ask questions about:

- AI Agents
- Prompt Engineering
- Research Papers
- General Knowledge
"""
)

# Session State Initialization

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(
        uuid.uuid4()
    )

# Display Previous Messages

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input

question = st.chat_input(
    "Ask a question..."
)

if question:

    # Display User Message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # LangGraph Config

    config = {
        "configurable": {
            "thread_id":
            st.session_state.thread_id
        }
    }

    # Invoke Graph

    response = app_graph.invoke(
        {
            "question": question,
            "chat_history":
            st.session_state.chat_history
        },
        config=config
    )

    answer = response["verified_answer"]

    sources = response.get(
        "sources",
        []
    )

    # Store Conversation History

    st.session_state.chat_history.append(
        {
            "user": question,
            "assistant": answer
        }
    )

    # Display Assistant Response

    with st.chat_message("assistant"):

        st.markdown(answer)

        if sources:

            st.markdown("### 📚 Sources")

            for source in sources:
                st.write(f"- {source}")

    # Save Assistant Message

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
    with st.sidebar:

        st.title("Agentic Research Assistant")

        st.markdown("""
        Features:

        ✅ Query Rewriting
        ✅ Intelligent Routing
        ✅ FAISS Retrieval
        ✅ Arxiv Search
        ✅ Wikipedia Search
        ✅ Cross Encoder Reranking
        ✅ Answer Verification
        ✅ Source Citations
        """)