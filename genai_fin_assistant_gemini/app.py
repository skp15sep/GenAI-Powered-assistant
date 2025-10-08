from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai_client import OpenAIClient
from processor import load_transactions_csv, answer_user_query
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="GenAI Financial Assistant")

# allow origins for development; tighten this for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "PUT"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    data_path: str = "sample_transactions.csv"

# health endpoint


@app.get("/health")
async def health():
    return {"status": "ok", "service": "GenAI Financial Assistant"}


@app.post("/query")
def query_financial_assistant(req: QueryRequest):
    try:
        df = load_transactions_csv(req.data_path)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to load data: {e}")

    try:
        openai_client = OpenAIClient()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = answer_user_query(openai_client, df, req.query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"openai processing failed: {e}")
