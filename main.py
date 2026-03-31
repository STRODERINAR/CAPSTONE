
from fastapi import FastAPI
from pydantic import BaseModel
import json

with open("data/data.json", "r", encoding="utf-8") as f:
    knowledge = json.load(f)

app = FastAPI()
# schema รับข้อความ
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    user_msg = request.message

    for item in knowledge:
        if item["question"] in user_msg:
            return {"answer": item["answer"]}

    return {"answer": "ไม่พบข้อมูล"}

