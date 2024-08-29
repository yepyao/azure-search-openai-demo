from abc import ABC
from typing import AsyncGenerator, Generator, List
from openai import AsyncOpenAI

from openai_messages_token_helper import build_messages, get_token_limit
from openai.types.chat import (
    ChatCompletion
)
import tiktoken

from .page import Page, SplitPage


class SemanticIndexer(ABC):

    def __init__(self, openai_client: AsyncOpenAI, chatgpt_model: str, chatgpt_deployment: str):
        self.openai_client = openai_client
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    async def get_related_content(self, search_content:str, full_content :str) -> str:
        tsg_check_prompt_template = """You are a assistant help get related content from a document which part is related to provided short content.
        User will first provide the short content, which is a search result by some keywords. 
        And then provide the full content followed by a seperator '---------------------'
        Your task is to determin, if user is interested in the provided short content, which part of the full content should also provided to the user.
        For example, is the short content is a symptom of a system failure, user should be interested in the solution to this failure.
        Firstly determin which part of the full content is related, and then summarize them in less than 500 tokens.
        Don't repeat the origin short content.
        If only no related content detected, response nothing.
        """

        user_query_request = f"{search_content}\n---------------------\n{full_content}"

        query_response_token_limit = 1000

        classification_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=tsg_check_prompt_template,
            new_user_content=user_query_request,
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

        return classification_completion.choices[0].message.content