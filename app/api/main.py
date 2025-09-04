from fastapi import FastAPI, HTTPException
from app.api.schemas import QueryRequest
from app.rag.pipeline import RAGPipeline
from app.core.logging import setup_logging
import faulthandler
import traceback

faulthandler.enable()
log = setup_logging()

app = FastAPI(title="Enterprise RAG Intelligence Hub")
pipeline = RAGPipeline()

@app.post("/query")
async def query(req: QueryRequest):
    try:
        print("inside query-", req.question)
        result = await pipeline.run(req.question)
        print("Final result keys:", list(result.keys()))
        return result
    except Exception as e:
        log.error("query_error", extra={"error": str(e)})
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal error")