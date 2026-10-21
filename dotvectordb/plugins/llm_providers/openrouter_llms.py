import os
from openai import AsyncOpenAI
from typing import Any, Dict, List, Tuple, Optional
from pydantic import BaseModel, Field
import json
from dotvectordb.interfaces.LanguageModel import LanguageModel, FinalResponse

class OpenRouterAgent(LanguageModel):
    def __init__(
        self,
        model_name: str = os.getenv("OPENROUTER_LLM_MODEL", "meta-llama/llama-3.1-8b-instruct"),
        api_key: str = os.getenv("OPENROUTER_API_KEY"),
        max_tokens: int = int(os.getenv("OPENROUTER_LLM_MAX_TOKENS", 4096)),
        Tokenizer: str = os.getenv("OPENROUTER_LLM_TOKENIZER", "gpt-4o"),
    ) -> None:
        print("Using OpenRouter LLM model:", model_name, "with max tokens:", max_tokens)
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            max_tokens=max_tokens,
            Tokenizer=Tokenizer,
        )
        self.client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=self.api_key,
    )
        
    async def call_llm(self,prompt, response_format=None, verbose=False):
            """
            Docstring generate
            Calls an LLM with the given prompt and returns the response in the specified format.
            Args:
                prompt (str): The prompt to send to the LLM.
                response_format: The desired response format (Pydantic model).
                model (str): The LLM model to use.
                max_tokens (int): The maximum number of tokens to generate.
            Returns:
                The LLM response parsed into the specified format.
            """
            if verbose:
                print("Prompt sent to LLM:", prompt)

            if response_format is not None:
                response = await self.client.beta.chat.completions.parse(   
                                model=self.model_name,
                                top_p=1,
                                max_tokens=self._max_tokens,
                                temperature=0.0,
                                frequency_penalty=0,    
                                presence_penalty=0,
                                extra_body={
                                    "provider": {
                                    "require_parameters": True,
                                    "allow_fallbacks": True
                                }
                                                                },
                                n=1,
                                response_format=response_format,
                                seed=123456,

                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant that helps people find information."},
                                    {"role": "user", "content": f"{prompt}"}
                                ],
                )
                llm_response = response.choices[0].message.content
                json_response = json.loads(llm_response)
            else:
                response = await self.client.beta.chat.completions.parse(   
                                model=self.model_name,
                                top_p=1,
                                max_tokens=self._max_tokens,
                                temperature=0.0,
                                frequency_penalty=0,    
                                presence_penalty=0,
                                n=1,
                                response_format=FinalResponse,
                                extra_body={
                                "provider": {
                                    "require_parameters": True,
                                    "allow_fallbacks": True
                                }
                                },
                                seed=123456,
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant that helps people find information."},
                                    {"role": "user", "content": f"{prompt}"}
                                ],
                )
                llm_response = response.choices[0].message.content
                json_response = json.loads(llm_response)
            return (
                      json_response,
                      response.usage.total_tokens,
                      response.usage.prompt_tokens,
                      response.usage.completion_tokens,
                  )
          
