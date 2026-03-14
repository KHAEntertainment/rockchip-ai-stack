# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict

import requests

logging.basicConfig(level=logging.INFO)


class CustomReranker:
    def __init__(self, reranking_endpoint: str):
        """
        Initialize the CustomReranker with the URL of an external reranking service.
        
        Parameters:
            reranking_endpoint (str): The HTTP endpoint (URL) of the reranking service to which rerank requests will be sent.
        """
        self._reranking_endpoint = reranking_endpoint
        logging.info(
            f"Initialized CustomReranker with reranking_endpoint: {self._reranking_endpoint}"
        )

    @property
    def reranking_endpoint(self) -> str:
        """
        Get the configured reranking service endpoint.
        
        Returns:
            str: The reranking endpoint URL.
        """
        return self._reranking_endpoint

    @reranking_endpoint.setter
    def reranking_endpoint(self, value: str):
        """
        Set the reranking service endpoint URL.
        
        Parameters:
            value (str): The full URL of the reranking HTTP endpoint.
        """
        self._reranking_endpoint = value

    def validate_retrieved_docs(self, retrieved_docs: Dict[str, Any]):
        """
        Validate that `retrieved_docs` includes the required keys 'question' and 'context'.
        
        Parameters:
            retrieved_docs (Dict[str, Any]): Mapping representing retrieved documents; must contain a 'question' entry and a 'context' entry (typically a list of document items).
        
        Raises:
            ValueError: If the 'question' key is missing.
            ValueError: If the 'context' key is missing.
        """
        if "question" not in retrieved_docs:
            raise ValueError("Question is required")
        if "context" not in retrieved_docs:
            raise ValueError("Context is required for reranker")

    def rerank(self, retrieved_docs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the input and either rerank the provided context or return the input unchanged.
        
        Parameters:
            retrieved_docs (Dict[str, Any]): Dictionary containing at minimum a "question" (str) and a "context" (list of document dicts). May optionally include a "history" key.
        
        Returns:
            Dict[str, Any]: If "context" contains items, a dictionary with the same "question", a reordered "context" (top results selected), and "history" preserved if present; otherwise the original `retrieved_docs` unchanged.
        
        Raises:
            ValueError: If `retrieved_docs` is missing required keys such as "question" or "context".
        """
        self.validate_retrieved_docs(retrieved_docs=retrieved_docs)
        if len(retrieved_docs["context"]) > 0:
            return self.rerank_tei(retrieved_docs=retrieved_docs)
        else:
            return retrieved_docs

    def rerank_tei(self, retrieved_docs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send the question and context texts to the configured reranking endpoint and return the documents reordered by the service's top scores.
        
        Parameters:
            retrieved_docs (Dict[str, Any]): Input mapping that must contain:
                - "question" (str): The query to rerank against.
                - "context" (List[Any]): A list of document-like objects where each item exposes `page_content` and corresponds by index to the reranker results.
                - "history" (optional): Any conversation history to be preserved.
        
        Returns:
            Dict[str, Any]: A dictionary with:
                - "question": the original question,
                - "context": a list of the top up-to-three documents from the original `context`, ordered by descending reranker score,
                - "history": the original history value if present, otherwise an empty string.
        
        Raises:
            Exception: If the HTTP response from the reranking endpoint has a non-200 status code; the exception message includes the status code and response text.
        """
        texts = [d.page_content for d in retrieved_docs["context"]]

        request_body = {
            "query": retrieved_docs["question"],
            "texts": texts,
            "raw_scores": False,
        }

        response = requests.post(
            url=f"{self.reranking_endpoint}",
            json=request_body,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        if response.status_code == 200:
            res = response.json()
            # Sort by score descending, pick top 3 or all if less than 3
            sorted_results = sorted(res, key=lambda x: x["score"], reverse=True)
            top_k = min(3, len(sorted_results))
            reranked_context = [
                retrieved_docs["context"][item["index"]] for item in sorted_results[:top_k]
            ]
            logging.info(
                f"Reranked context for question '{retrieved_docs['question']}': "
                f"{reranked_context}"
            )

            return {
                "question": retrieved_docs["question"],
                "context": reranked_context,
                "history": retrieved_docs.get("history", ""),
            }
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
