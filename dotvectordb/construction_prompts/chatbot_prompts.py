from typing import Optional

from pydantic import BaseModel, Field


class ChatbotRetrievalDecision(BaseModel):
    needs_retrieval: bool = Field(
        description=(
            "Whether external retrieval is required to answer the latest "
            "user message accurately."
        )
    )
    search_query: Optional[str] = Field(
        default=None,
        description=(
            "A concise standalone retrieval query when needs_retrieval is "
            "true; otherwise null."
        ),
    )


CHATBOT_ROUTING_PROMPT = """
You route messages for a conversational RAG assistant.

Decide whether the latest user message requires searching the selected
retrieval system. Retrieval is useful for questions that depend on facts,
documents, entities, relationships, or other information that may exist in
the user's knowledge base. Retrieval is unnecessary for greetings, casual
conversation, writing transformations that already include all necessary
text, or questions answerable entirely from the conversation.

If retrieval is needed, create one concise standalone search query. Resolve
pronouns and references using the conversation history. Do not answer the
question. If retrieval is not needed, set search_query to null.

Conversation history:
{conversation_history}

Latest user message:
{user_message}
""".strip()


CHATBOT_RESPONSE_PROMPT = """
Answer the latest user message as a helpful conversational assistant. Use the
conversation history for continuity. When retrieved context is provided,
ground the answer in that context. If the retrieved context does not contain
the requested information, say so rather than inventing facts.

Conversation history:
{conversation_history}

Latest user message:
{user_message}

Retrieved context:
{retrieved_context}

Assistant response:
""".strip()

