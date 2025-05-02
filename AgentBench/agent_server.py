# agent_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from main import LangGraphAgent   # your agent class

app = FastAPI()
agent = LangGraphAgent()          # initialize it once

class Query(BaseModel):
    prompt: str

@app.post("/predict")
def predict(q: Query):
    bash_script = agent.run(q.prompt)
    return {"bash": bash_script}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
