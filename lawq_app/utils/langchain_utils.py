from operator import itemgetter
from typing import Any, Dict, List

from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.messages import AIMessage


class DocAnswerChain:
    """
    Pehle yahan ChatOpenAI use ho raha tha.
    Ab hum sirf retriever ka context leke ek local/demo answer bana rahe hain,
    taaki OpenAI API / quota ki zaroorat na pade.
    """

    def __init__(self) -> None:
        self.init_prompt = (
            "You are a helper for answering questions about legal contracts "
            "using ONLY the provided context. "
            "This app is running in offline/demo mode (no OpenAI API). "
            "So answer by quoting and summarizing the retrieved snippets only."
        )

    def _build_answer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        docs = inputs.get("contract_context") or []
        question = inputs.get("query_prompt", "")

        # Docs ko plain text me convert karo
        if isinstance(docs, list):
            parts: List[str] = []
            for i, d in enumerate(docs, 1):
                text = getattr(d, "page_content", str(d))
                parts.append(f"[{i}] {text}")
            context_text = "\n\n".join(parts)
        else:
            context_text = str(docs)

        if not context_text.strip():
            content = (
                self.init_prompt
                + "\n\n"
                + "No relevant context was retrieved for this question, "
                  "so I cannot answer it based on the documents.\n\n"
                + f"Question: {question}"
            )
        else:
            content = (
                self.init_prompt
                + "\n\n"
                + f"Question:\n{question}\n\n"
                  "Here are the most relevant contract snippets I found:\n\n"
                + context_text
            )

        # main.py expect karta hai: result['response'].content
        return {
            "response": AIMessage(content=content),
            "context": docs,
        }

    def generate_response_chain(self, retriever_base):
        """
        retriever_base: Chroma retriever (VectorStoreHandler se aata hai)
        Input:  {"query_prompt": "user ka question"}
        Output: {"response": AIMessage, "context": docs}
        """

        # Pehle retriever se context nikalte hain
        parallel_retriever = RunnableParallel(
            {
                "contract_context": itemgetter("query_prompt") | retriever_base,
                "query_prompt": itemgetter("query_prompt"),
            }
        )

        # Fir simple local answer banate hain
        qa_chain = parallel_retriever | RunnableLambda(self._build_answer)
        return qa_chain
