# test_ollama_2.py
import asyncio
from app.models.llm_ollama import OllamaLLM

async def test_llm():
    llm = OllamaLLM()
    prompt = (
        "You are an enterprise assistant.\n"
        "Answer only using the provided context. If the context does not contain the answer, reply exactly:\n"
        "\"I don't have enough information in the knowledge base to answer that.\"\n"
        "Do not use prior knowledge or assumptions. Cite sources as [doc:source_id] after each sentence.\n\n"
        "Context:\n"
        "- Self-care activities and strategies are brought to the table, examined, and shortlisted.\n"
        "- Assessment is key to building a successful routine.\n"
        "- The Self-Care Wheel includes six dimensions and 88 strategies.\n\n"
        "Question: What are the components of a self-care routine?"
    )
    response = await llm.generate(prompt)
    print("🔍 LLM Response:\n", response)

if __name__ == "__main__":
    asyncio.run(test_llm())