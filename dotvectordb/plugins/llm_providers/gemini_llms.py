import os
from typing import Any, Dict, List, Tuple, Optional
from pydantic import BaseModel, Field
import json
from dotvectordb.interfaces.LanguageModel import LanguageModel, FinalResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional

class GeminiAgent(LanguageModel):
    def __init__(
        self,
        model_name: str = os.getenv("GEMINI_LLM_MODEL", "gemini-3.5-flash"),
        api_key: str = os.getenv("GEMINI_API_KEY"),
        max_tokens: int = int(os.getenv("GEMINI_LLM_MAX_TOKENS", 4096)),
        Tokenizer: str = os.getenv("OPENAI_LLM_TOKENIZER", "gpt-4o"),
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            max_tokens=max_tokens,
            Tokenizer=Tokenizer,
        )
        self.client = genai.Client()

    #############################################################################################
    #  LLM call Function                                                                       #                                                          
    #############################################################################################
    # These functions handle calls to the LLM for various purposes.
    # If you'll like to add more LLMs, edit these functions accordingly.
    ############################################################################################
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
            response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful assistant that helps people find information.",
                top_p=1,
                max_output_tokens=self._max_tokens,
                temperature=0.0,
                frequency_penalty=0,
                presence_penalty=0,
                candidate_count=1,
                seed=123456,
                response_mime_type="application/json",
                response_schema=response_format,  # FinalResponse in the else branch
            ),
        )
            llm_response = response.text
            json_response = response.parsed.model_dump()
        else:
            response = await self.client.aio.models.generate_content(
                            model=self.model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction="You are a helpful assistant that helps people find information.",
                                top_p=1,
                                max_output_tokens=self._max_tokens,
                                temperature=0.0,
                                frequency_penalty=0,
                                presence_penalty=0,
                                candidate_count=1,
                                seed=123456,
                                response_mime_type="application/json",
                                response_schema=FinalResponse,
                            ),
                        )
            llm_response = response.text
            json_response = response.parsed.model_dump()
        tokens = self.tokenize(llm_response)
        return json_response, len(tokens)
