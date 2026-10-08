from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    url: str = ""
    chat_history: list = []


class FeedbackRequest(BaseModel):
    feedback: str
    comment: str = ""