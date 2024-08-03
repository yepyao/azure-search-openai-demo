import asyncio
from io import BytesIO
import logging
import re
from tokenize import Whitespace
import urllib
from typing import Any, Coroutine, List, Literal, Optional, Union, overload

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai_messages_token_helper import build_messages, get_token_limit
import tiktoken

from text import nonewlines
from approaches.approach import ThoughtStep
from approaches.chatapproach import ChatApproach
from core.authentication import AuthenticationHelper
from azure.storage.blob.aio import ContainerClient


class ChatAKSArcTSGApproach(ChatApproach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then uses Azure AI Search to retrieve relevant documents, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
        blob_container_client:ContainerClient
    ):
        self.search_client = search_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)
        self.blob_container_client = blob_container_client

    @property
    def system_message_chat_conversation(self):
        return """Assistant helps the AKS Arc developer and support engineers with their knowledge related questions, and questions about how to troubleshooting and solve a software issue. Be brief in your answers.
        AKS Arc is a software production from Microsoft, running Kubernetes clusters on hybrid envrionemnts. It also known as Hybrid AKS, AKS Hybrid.
        Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
        For tabular information return it as an html table. Do not return markdown format. If the question is not in English, answer in the language used in the question.
        Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, for example [info1.txt]. Don't combine sources, list each source separately, for example [info1.txt][info2.pdf].
        {follow_up_questions_prompt}
        {injected_prompt}
        """

    def truncate_if_exceed_max_token(self, text, max_tokens):
        tokenizer = tiktoken.get_encoding("cl100k_base")

        # Tokenize the text
        encoding = tokenizer.encode(text)

        # Check if the number of tokens exceeds max_tokens
        if len(encoding) > max_tokens:
            # Truncate the text
            truncated_ids = encoding.ids[:max_tokens]
            # Decode the truncated tokens back to text
            truncated_text = tokenizer.decode(truncated_ids)
            return truncated_text
        else:
            return text
    
    async def get_tsg_summary_list_await(
        self,
        related_doc: str,
    ) -> str:
        blob = await self.blob_container_client.get_blob_client(related_doc).download_blob()
        stream = BytesIO()
        await blob.readinto(stream)
        blob_content = stream.getvalue().decode("utf-8")
        # Based on the origin documents to generate the solution
        print(f"Download doc: {related_doc} Length: {len(blob_content)}")

        # summary the content
        # TODO: summary can be done in index stage
        tsg_summary_template = """You are an assistant helping to summarize AKS Arc production-related Troubleshooting Guides (TSGs).

        A Troubleshooting Guide (TSG) is a document that outlines known issues of a software product and provides respective solutions. Each TSG typically includes:

        Symptom/Observation: Describes the production issue, often including an error message returned by the system or a description of abnormal system behavior.
        Issue Validation: Guides the user on how to confirm if the described issue matches the one they are experiencing.
        Root Cause and Solution: Explains the root cause of the issue and provides a solution, often with step-by-step instructions.
        Your task is to summarize the TSG document provided by the user. Your summary should include:

        A precise description of the issue, retaining any error messages.
        A brief statement about the cause of the issue.
        A summary of the solution, including steps if applicable, in less than 400 words.
        The total response should be less than 800 words.
        """

        query_response_token_limit = 1000
        tsg_summary_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=tsg_summary_template,
            new_user_content= self.truncate_if_exceed_max_token(blob_content, self.chatgpt_token_limit - query_response_token_limit - 1000),
            max_tokens=self.chatgpt_token_limit - query_response_token_limit,
        )

        tsg_summary_completion: ChatCompletion = await self.openai_client.chat.completions.create(
            messages=tsg_summary_messages,  # type: ignore
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            temperature=0.0,  # Minimize creativity for question classification
            max_tokens=query_response_token_limit,  # Setting too low risks malformed JSON, setting too high may affect performance
            n=1,
        )

        tsg_summary_result = tsg_summary_completion.choices[0].message.content
        # update content to provide
        content = related_doc+": "+nonewlines(tsg_summary_result or "")
        print(f"Summary of related doc: {content}")
        return content

    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[False],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, ChatCompletion]]: ...

    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[True],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, AsyncStream[ChatCompletionChunk]]]: ...

    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]]]:
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top", 10)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        filter = self.build_filter(overrides, auth_claims)

        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")
        user_query_request = "Generate search query for: " + original_user_query

        tools: List[ChatCompletionToolParam] = [
            {
                "type": "function",
                "function": {
                    "name": "search_sources",
                    "description": "Retrieve sources from the Azure AI Search index",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "Query string to retrieve documents from azure search eg: 'Health care plan'",
                            }
                        },
                        "required": ["search_query"],
                    },
                },
            }
        ]

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        query_response_token_limit = 100
        query_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=self.query_prompt_template,
            tools=tools,
            few_shots=self.query_prompt_few_shots,
            past_messages=messages[:-1],
            new_user_content=user_query_request,
            max_tokens=self.chatgpt_token_limit - query_response_token_limit,
        )

        chat_completion: ChatCompletion = await self.openai_client.chat.completions.create(
            messages=query_messages,  # type: ignore
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            temperature=0.0,  # Minimize creativity for search query generation
            max_tokens=query_response_token_limit,  # Setting too low risks malformed JSON, setting too high may affect performance
            n=1,
            tools=tools,
        )

        query_text = self.get_search_query(chat_completion, original_user_query)

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if use_vector_search:
            vectors.append(await self.compute_text_embedding(query_text))

        results = await self.search(
            top,
            query_text,
            filter,
            vectors,
            use_text_search,
            use_vector_search,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
        )

        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)
        content = "\n".join(sources_content)

        # STEP 3: Check if original_user_query is asking a trouble shooting question and the which source documents describe the same issue.

        tsg_check_prompt_template = """You are a assistant help with AKS Arc production related questions. User may ask some knowledge question, or want to get solution for a prodcution issue.
        User will provide a question with some source docs. The source doc will start in a new line, with its doc name first, and then its content after a ':'.
        You need to judge whether the question is asking for a solution for a production error or issue. Answer with 'Yes' or 'No' in the first line.
        If the answer is yes, you need to judge which source doc describe the same issue with the user question. If no same issue, select one doc which is most similar with the question.
        Deduplicate the related doc and sort them from most related to less related.
        Reply with the source doc names, MUST in format '<<doc name>>', keep the doc name as it is, each in one line.
        """

        classification_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=tsg_check_prompt_template,
            new_user_content=original_user_query + "\n\nSources:\n" + content,
            max_tokens=self.chatgpt_token_limit - query_response_token_limit,
        )

        classification_completion: ChatCompletion = await self.openai_client.chat.completions.create(
            messages=classification_messages,  # type: ignore
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            temperature=0.0,  # Minimize creativity for question classification
            max_tokens=query_response_token_limit,  # Setting too low risks malformed JSON, setting too high may affect performance
            n=1,
        )

        classification_result = classification_completion.choices[0].message.content

        if (classification_result.lower().startswith('yes')):
            # Get the whole contents of origin documents
            related_docs = re.findall(r"<<([^>>]+)>>", classification_result)
            print(f"related_docs: {related_docs}")
            
            tsg_summary_list_await = []
            for related_doc in related_docs:
                tsg_summary_list_await.append(asyncio.create_task(self.get_tsg_summary_list_await(related_doc)))

            content = ""
            for tsg_summary_await in tsg_summary_list_await:
                content += "\n" + await tsg_summary_await

        # STEP n: Generate a contextual and content specific answer using the search results and chat history

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        system_message = self.get_system_prompt(
            overrides.get("prompt_template"),
            self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else "",
        )

        response_token_limit = 1024
        messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=system_message,
            past_messages=messages[:-1],
            # Model does not handle lengthy system messages well. Moving sources to latest user conversation to solve follow up questions prompt.
            new_user_content=original_user_query + "\n\nSources:\n" + content,
            max_tokens=self.chatgpt_token_limit - response_token_limit,
        )

        data_points = {"text": sources_content}

        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Prompt to generate search query",
                    [str(message) for message in query_messages],
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
                ThoughtStep(
                    "Search using generated search query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "top": top,
                        "filter": filter,
                        "use_vector_search": use_vector_search,
                        "use_text_search": use_text_search,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
                ThoughtStep(
                    "Prompt to classify if the search query is a TSG search query and rank the related docs",
                    [str(message) for message in classification_messages],
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
                ThoughtStep(
                    "Classification result: TSG search query?",
                    classification_result,
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    [str(message) for message in messages],
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
            ],
        }

        chat_coroutine = self.openai_client.chat.completions.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=response_token_limit,
            n=1,
            stream=should_stream,
        )
        return (extra_info, chat_coroutine)
