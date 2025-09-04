# tests/test_pipeline.py
import asyncio
from app.rag.pipeline import RAGPipeline

async def main():
    pipe = RAGPipeline()
    out = await pipe.run("Summarize the PTO policy and notice period.")
    assert isinstance(out["answer"], str)
    assert "confidence" in out
    print(out)

if __name__ == "__main__":
    asyncio.run(main())
