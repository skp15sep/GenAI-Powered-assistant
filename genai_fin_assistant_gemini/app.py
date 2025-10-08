from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai_client import OpenAIClient
from processor import load_transactions_csv, answer_user_query

app = FastAPI(title="GenAI Financial Assistant (Gemini)")


class QueryRequest(BaseModel):
    api_key: str | None = None
    query: str
    data_path: str = "sample_transactions.csv"


@app.post("/query")
def query_financial_assistant(req: QueryRequest):
    try:
        df = load_transactions_csv(req.data_path)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to load data: {e}")

    try:
        # gemini_client = GeminiClient(api_key=req.api_key)
        openai_client = OpenAIClient(api_key=req.api_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        # result = answer_user_query(gemini_client, df, req.query)
        result = answer_user_query(openai_client, df, req.query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"openai processing failed: {e}")
