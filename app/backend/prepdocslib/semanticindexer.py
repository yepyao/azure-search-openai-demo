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
        self.encoding = tiktoken.get_encoding(chatgpt_model)

    def index(self, pages: List[Page]) -> AsyncGenerator[SplitPage, None]:
        for page in pages:
            if len(page.text) > 0:
                token_num = len(self.encoding.encode(page.text))
                if token_num < 5000:

                for section in sections:
                    print("-----new section----")
                    print(section)
                    # tsg_check_prompt_template = """You are a assistant help with AKS Arc production related questions. User may ask some knowledge question, or want to get solution for a prodcution issue.
                    # User will provide a question with some source docs. The source doc will start in a new line, with its doc name first, and then its content after a ':'.
                    # You need to judge whether the question is asking for a solution for a production error or issue. Answer with 'Yes' or 'No' in the first line.
                    # If the answer is yes, you need to judge which source doc describe the same issue with the user question. If no same issue, select one doc which is most similar with the question.
                    # Deduplicate the related doc and sort them from most related to less related.
                    # Reply with the source doc names, in format '<<doc name>>', each in one line.
                    # """

                    # query_response_token_limit = 1000

                    # classification_messages = build_messages(
                    #     model=self.chatgpt_model,
                    #     system_prompt=tsg_check_prompt_template,
                    #     new_user_content=content,
                    #     max_tokens=self.chatgpt_token_limit - query_response_token_limit,
                    # )

                    # classification_completion: ChatCompletion = await self.openai_client.chat.completions.create(
                    #     messages=classification_messages,  # type: ignore
                    #     # Azure OpenAI takes the deployment name as the model name
                    #     model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
                    #     temperature=0.0,  # Minimize creativity for question classification
                    #     max_tokens=query_response_token_limit,  # Setting too low risks malformed JSON, setting too high may affect performance
                    #     n=1,
                    # )

                    # classification_result = classification_completion.choices[0].message.content