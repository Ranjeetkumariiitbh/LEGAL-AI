import streamlit as st
from dotenv import load_dotenv

from htmlTemplates import css, bot_template

from utils.doc_utils import PDFHandler
from utils.text_split_utils import TextChunker
from utils.vector_db_utils import VectorStoreHandler
from utils.langchain_utils import DocAnswerChain

# Load .env (not required in offline mode, but fine to keep)
load_dotenv(override=True)


def get_conversation_chain(retriever):
    my_lang_chain = DocAnswerChain()
    return my_lang_chain.generate_response_chain(retriever_base=retriever)


def handle_userinput(user_question: str):
    if not st.session_state.conversation:
        st.error("Please upload documents and click Process first.")
        return

    result = st.session_state.conversation.invoke(
        {"query_prompt": user_question}
    )

    answer = result["response"].content
    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Contract AI", page_icon="🤖", layout="wide")
    st.write(css, unsafe_allow_html=True)

    # ===== Profile Banner / Title =====
    st.markdown(
        """
        <div style="text-align:center; margin-top: 10px; margin-bottom: 20px;">
            <h2 style="margin-bottom: 4px;">Ranjeet Kumar</h2>
            <p style="margin-top: 0; font-size: 16px; opacity: 0.85;">IIIT Bhagalpur</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # ===== Main Title / Description =====
    st.title("📄 Contract AI Assistant")
    st.write("Upload your contract PDFs and ask questions based on the document content.")

    user_question = st.text_input("What is your question?")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.markdown("### 👤 Ranjeet Kumar (IIIT Bhagalpur)")
        st.caption("Contract AI • PDF-based Q&A (Offline Demo Mode)")

        with st.expander("📌 How to use", expanded=True):
            st.markdown(
                """
                1) PDFs upload karo  
                2) **Process** click karo  
                3) Question box me apna sawal likho  
                """
            )

        with st.expander("💡 Sample Questions"):
            st.markdown(
                """
                - Termination clause kya hai?  
                - Notice period kitna hai?  
                - Payment terms & due date kya hai?  
                - Late payment penalty mention hai kya?  
                - Confidentiality / NDA clause ka summary?  
                - Renewal / extension condition kya hai?  
                """
            )

        with st.expander("🔒 Notes"):
            st.markdown(
                """
                - Ye version **offline/demo** mode me hai (OpenAI quota issue avoid karne ke liye).  
                - Answers **retrieved contract snippets** ke basis pe aate hain.  
                """
            )

        st.divider()
        st.subheader("Your documents")

        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf"],
        )

        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
                return

            with st.spinner("Processing..."):
                # 1) Extract text from PDFs
                pdf = PDFHandler(pdf_files=pdf_docs)
                raw_text = pdf.extract_text_from_pdfs()

                if not raw_text.strip():
                    st.error("Could not extract any text from the uploaded PDFs.")
                    return

                # 2) Split into chunks
                text_splitter = TextChunker(raw_text)
                text_chunks = text_splitter.split_into_chunks()

                if not text_chunks:
                    st.error("Text splitting produced no chunks.")
                    return

                # 3) Build vector store (OFFLINE embeddings)
                my_vector_store = VectorStoreHandler()
                my_vector_store.create_embeddings(text_chunks)
                retriever = my_vector_store.get_retriever()

                # 4) Build conversation chain
                st.session_state.conversation = get_conversation_chain(retriever)

            st.success("Done! Now ask your question above.")


if __name__ == "__main__":
    main()
