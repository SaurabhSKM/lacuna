"""
Thin wrapper around whichever LLM provider is configured. The point of
this file is that every other module just calls get_llm() and never
cares whether it's Groq, Gemini, or something else later - that's the
"model-agnostic" property LangChain's chat model interface gives us.
"""
from src.config import config


class MissingAPIKeyError(RuntimeError):
    pass


def get_llm():
    provider = config.LLM_PROVIDER.lower()

    if provider == "groq":
        if not config.GROQ_API_KEY:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com and add it to your .env file."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_MODEL,
            temperature=0.1,
        )

    if provider == "gemini":
        if not config.GOOGLE_API_KEY:
            raise MissingAPIKeyError(
                "GOOGLE_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and add it to your "
                ".env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=config.GOOGLE_API_KEY,
            model=config.GEMINI_MODEL,
            temperature=0.1,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Use 'groq' or 'gemini'."
    )


def invoke_structured(llm, schema, prompt: str, max_retries: int = 2, callbacks: list = None):
    """
    Bind `schema` (a Pydantic model) as structured output on `llm`, invoke
    `prompt`, and retry up to max_retries times if the response fails
    schema validation.

    This is shared by every node that needs a validated object back from
    the LLM (the Answerer, the Completeness-checker, and later the
    Enrichment node) - one retry mechanism instead of each one
    reimplementing its own, which is exactly the kind of duplication that
    tends to drift out of sync over time.

    `callbacks`, if provided, is passed through to .invoke() - this is
    how Langfuse tracing (src/tracing.py) hooks into every structured
    call without this function needing to know tracing exists.

    Raises RuntimeError if every attempt fails, rather than returning None -
    callers should always be prepared to catch this.
    """
    structured_llm = llm.with_structured_output(schema)
    invoke_kwargs = {"config": {"callbacks": callbacks}} if callbacks else {}

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return structured_llm.invoke(prompt, **invoke_kwargs)
        except Exception as e:  # validation errors, malformed tool calls, etc.
            last_error = e
    raise RuntimeError(
        f"Structured output ({schema.__name__}) failed after "
        f"{max_retries} attempt(s): {last_error}"
    ) from last_error
