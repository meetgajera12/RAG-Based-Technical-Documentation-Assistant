import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Technical Documentation Assistant",
    page_icon="📚"
)


st.title("Technical Documentation Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = []


st.sidebar.title("Document Management")


uploaded_file = st.sidebar.file_uploader(
    "Upload PDF, TXT or MD",
    type=["pdf", "txt", "md"]
)


if st.sidebar.button("Upload Document"):

    if uploaded_file is None:

        st.sidebar.warning("Please select a file.")

    else:

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        with st.sidebar.spinner("Generating embeddings..."):
            response = requests.post(
                f"{API_URL}/ingest",
                files=files
            )

        if response.status_code == 200:

            data = response.json()

            st.sidebar.success(
                f"Uploaded successfully. "
                f"Chunks: {data['chunks']}"
            )

        else:

            st.sidebar.error(response.text)


st.sidebar.divider()


url = st.sidebar.text_input(
    "Add document URL (Optional)"
)


if st.sidebar.button("Add URL"):

    if not url:

        st.sidebar.warning("Enter a URL.")

    else:

        response = requests.post(
            f"{API_URL}/ingest",
            params={"url": url}
        )

        if response.status_code == 200:

            data = response.json()

            st.sidebar.success(
                f"URL added. "
                f"Chunks: {data['chunks']}"
            )

        else:

            st.sidebar.error(response.text)


st.sidebar.divider()


if st.sidebar.button("Show Documents"):

    response = requests.get(
        f"{API_URL}/documents"
    )

    if response.status_code == 200:

        data = response.json()

        if data["documents"]:

            st.sidebar.write("Indexed documents:")

            for document in data["documents"]:

                st.sidebar.write(
                    f"- {document}"
                )

        else:

            st.sidebar.info(
                "No documents indexed."
            )

    else:

        st.sidebar.error(response.text)


st.sidebar.divider()


if st.sidebar.button("Clear Chat"):

    st.session_state.messages = []

    st.rerun()


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


question = st.chat_input(
    "Ask a question..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.write(question)


    chat_history = st.session_state.messages[:-1]


    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question": question,
                    "url": "",
                    "chat_history": chat_history
                }
            )


        if response.status_code == 200:

            data = response.json()

            answer = data["answer"]

            st.write(answer)

            if data.get("chunks"):

                with st.expander("View relevant chunks"):

                    for i, chunk in enumerate(data["chunks"], 1):
                        st.write(f"**Chunk {i}**")
                        st.caption(f"Source: {chunk['source']}")
                        st.write(chunk["content"])
                        st.divider()

            st.caption(
                f"Search method: {data['source']}"
            )

            if data["sources"]:

                st.write("Sources:")

                for source in data["sources"]:

                    st.write(
                        f"- {source}"
                    )


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


            st.session_state.last_answer = answer


        else:

            st.error(response.text)


if "last_answer" in st.session_state:

    st.divider()

    st.write("Was this answer helpful?")

    col1, col2 = st.columns(2)


    with col1:

        if st.button("👍 Yes"):

            response = requests.post(
                f"{API_URL}/feedback",
                json={
                    "feedback": "up",
                    "comment": ""
                }
            )

            if response.status_code == 200:

                st.success("Thanks for your feedback.")


    with col2:

        if st.button("👎 No"):

            response = requests.post(
                f"{API_URL}/feedback",
                json={
                    "feedback": "down",
                    "comment": ""
                }
            )

            if response.status_code == 200:

                st.success("Thanks for your feedback.")