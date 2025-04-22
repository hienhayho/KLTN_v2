from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.flow import AppFlow

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"ping": "pong"}


class QueryRequest(BaseModel):
    query: str
    history: list[str] = []


@app.post("/query")
async def query(query_request: QueryRequest):
    """
    Process the input query and return the result.

    Args:
        query (str): The input query.

    Returns:
        dict: The result of processing the query.
    """
    flow = AppFlow(timeout=1000, verbose=False)
    res = await flow.run(query=query_request.query, history=query_request.history)

    return {"answer": res.answer, "final_query": res.final_query}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
