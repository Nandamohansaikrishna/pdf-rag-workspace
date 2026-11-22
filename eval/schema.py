from typing import Literal, Optional
from pydantic import BaseModel, model_validator

class EvalItem(BaseModel):
    id: str
    question: str
    type: Literal['easy', 'multi_hop', 'unanswerable']
    split: Literal['dev', 'test']
    answerable: bool
    reference_answer: Optional[str] = None
    doc: Optional[str] = None
    pages: list[int] = []

    @model_validator(mode='after')
    def check_fields(self):
        if self.answerable and not (self.reference_answer and self.doc and self.pages):
            raise ValueError('answerable items need reference_answer, doc and pages')
        if self.answerable == (self.type == 'unanswerable'):
            raise ValueError('type and answerable disagree')
        return self