from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: list[str]

class ClassifyInput(BaseModel):
    text: str
    labels: list[str] = ["politics", "technology", "sports", "entertainment", "business", "science", "health"]
