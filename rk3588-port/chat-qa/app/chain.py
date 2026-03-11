# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import logging

import lancedb
from langchain_community.vectorstores import LanceDB as LangChainLanceDB
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_openai import ChatOpenAI as EGAIModelServing
from langchain_openai import OpenAIEmbeddings as EGAIEmbeddings
from shared.lancedb_schema import get_or_create_table
from .custom_reranker import CustomReranker

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Settings from environment
# ---------------------------------------------------------------------------

LANCEDB_PATH = os.getenv("LANCEDB_PATH", "./data/lancedb")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
EMBEDDING_ENDPOINT_URL = os.getenv("EMBEDDING_ENDPOINT_URL", "http://localhost:8001/v1")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Qwen2.5-VL")
FETCH_K = int(os.getenv("FETCH_K", "10"))

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

try:
    embedder = EGAIEmbeddings(
        openai_api_key="EMPTY",
        openai_api_base="{}".format(EMBEDDING_ENDPOINT_URL),
        model=MODEL_NAME,
    )
    logging.info(
        "Embeddings initialized with endpoint configured in EMBEDDING_ENDPOINT_URL"
    )
except Exception as e:
    logging.error(f"Failed to initialize embeddings: {str(e)}")
    raise

# ---------------------------------------------------------------------------
# LanceDB vector store and retriever
# ---------------------------------------------------------------------------

db = lancedb.connect(LANCEDB_PATH)
table = get_or_create_table(db, COLLECTION_NAME)
knowledge_base = LangChainLanceDB(
    connection=db,
    embedding=embedder,
    table_name=COLLECTION_NAME,
)
retriever = knowledge_base.as_retriever(
    search_type="mmr",
    search_kwargs={"k": FETCH_K},
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

template = """
Use the following pieces of context from retrieved
dataset and prior conversation history to answer the question.
Do not make up an answer if there is no context provided to help answer it.

Conversation history:
---------
{history}
---------

Context:
---------
{context}

---------
Question: {question}
---------

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# ---------------------------------------------------------------------------
# LLM backend detection
# ---------------------------------------------------------------------------

ENDPOINT_URL = os.getenv("LLM_ENDPOINT_URL", "http://localhost:8080/v1")
LLM_BACKEND_ENV = os.getenv("LLM_BACKEND", "").lower()

# Determine backend from explicit env var first, then fall back to URL heuristic.
if LLM_BACKEND_ENV:
    LLM_BACKEND = LLM_BACKEND_ENV
elif "ovms" in ENDPOINT_URL.lower():
    LLM_BACKEND = "ovms"
elif "text-generation" in ENDPOINT_URL.lower():
    LLM_BACKEND = "tgi"
elif "vllm" in ENDPOINT_URL.lower():
    LLM_BACKEND = "vllm"
else:
    LLM_BACKEND = "unknown"

logging.info(f"Using LLM inference backend: {LLM_BACKEND}")

LLM_MODEL = os.getenv("LLM_MODEL_NAME", "Qwen2.5-VL")
RERANKER_ENDPOINT = os.getenv("RERANKER_ENDPOINT_URL", "http://localhost:8003") + "/rerank"
callbacks = [StreamingStdOutCallbackHandler()]

# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

async def context_retriever_fn(chain_inputs: dict):
    """
    Retrieve relevant documents for a given question and conversation history.

    Args:
        chain_inputs (dict): Dictionary with "question" and "history" keys.

    Returns:
        list: List of relevant Document objects (may be empty if no question or no results).

    Raises:
        ValueError: If chain_inputs is not a dict.
    """
    if not isinstance(chain_inputs, dict):
        raise ValueError("Invalid input: chain_inputs must be a dictionary.")

    # in process_chunks we already raise ValueError for empty question, but to keep
    # shape consistent we return empty list here
    question = chain_inputs.get("question", "")
    if not question:
        return {}  # to keep shape consistent

    retrieved_docs = await retriever.aget_relevant_documents(question)
    return retrieved_docs  # context: list[Document]


def format_docs(docs):
    """
    Format a list of Document objects into a readable string for prompt context.

    Args:
        docs (list): List of Document objects (each with .page_content and .metadata attributes).

    Returns:
        str: Formatted string with each document's sl.no, content and source, or a message if
        no docs are found.
    """
    if not docs:
        return "No relevant context found."

    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        metadata = doc.metadata or {}
        source = metadata.get("source", "Unknown source")

        formatted_docs.append(f"[Document {i}] {content}\nSource: {source}")

    return "\n\n".join(formatted_docs)

# ---------------------------------------------------------------------------
# Main streaming entry-point
# ---------------------------------------------------------------------------

async def process_chunks(conversation_messages, max_tokens):
    """
    Process a list of conversation messages and stream the LLM-generated answer.

    This function builds the retrieval-augmented generation (RAG) chain, including context
    retrieval, reranking, prompt formatting, and LLM inference. It streams the output as
    server-sent events.

    Args:
        conversation_messages (list): List of message objects, each with 'role' and 'content'.
            The last message is treated as the user's question.
        max_tokens (int): Maximum number of tokens for the LLM response (if supported by
            backend).

    Yields:
        str: Server-sent event strings ("data: ...\\n\\n") with the LLM's output chunks.

    Raises:
        ValueError: If the question text is empty or only whitespace.
    """
    # All messages except the last one are considered history
    if len(conversation_messages) > 1:
        valid_history_msgs = []
        for msg in conversation_messages[:-1]:
            if hasattr(msg, "role") and hasattr(msg, "content") and msg.content is not None:
                valid_history_msgs.append(f"{msg.role}: {msg.content}")
            else:
                logging.warning("Skipping history message: missing 'role' or 'content'")
        history = "\n".join(valid_history_msgs)
    else:
        history = ""

    # Raise ValueError if question is empty or only whitespace
    question_text = conversation_messages[-1].content
    if not question_text or not question_text.strip():
        raise ValueError("Question text cannot be empty")

    context_retriever = RunnableLambda(context_retriever_fn)

    # "llama" and "vllm" both use the OpenAI-compatible endpoint without a seed parameter.
    # "tgi", "ovms", and other backends pass seed + max_tokens.
    if LLM_BACKEND in ["vllm", "llama", "unknown"]:
        model = EGAIModelServing(
            openai_api_key="EMPTY",
            openai_api_base="{}".format(ENDPOINT_URL),
            model_name=LLM_MODEL,
            top_p=0.99,
            temperature=0.01,
            streaming=True,
            callbacks=callbacks,
            stop=["\n\n"],
        )
    else:
        seed_value = int(os.getenv("SEED", 42))
        model = EGAIModelServing(
            openai_api_key="EMPTY",
            openai_api_base="{}".format(ENDPOINT_URL),
            model_name=LLM_MODEL,
            top_p=0.99,
            temperature=0.01,
            streaming=True,
            callbacks=callbacks,
            seed=seed_value,
            max_tokens=max_tokens,
        )

    re_ranker = CustomReranker(reranking_endpoint=RERANKER_ENDPOINT)
    re_ranker_lambda = RunnableLambda(re_ranker.rerank)

    # RAG Chain
    chain = (
        RunnableParallel({
            "context": context_retriever,
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
        })
        | re_ranker_lambda
        | {
            "context": (lambda x: format_docs(x["context"])),
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
        }
        | prompt
        | model
        | StrOutputParser()
    )

    chain_input = {
        "history": history,
        "question": question_text,
    }

    async for log in chain.astream(chain_input):
        yield f"data: {log}\n\n"
