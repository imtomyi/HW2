from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: list[str]

class SummarizeInput(BaseModel):
    text: str
    max_length: int = 130
    min_length: int = 30
