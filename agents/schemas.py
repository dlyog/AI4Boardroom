from pydantic import BaseModel, Field

class DocumentClassification(BaseModel):
    """
    Schema for the output of our document‐classifier agent.
    """
    document_type: str = Field(
        ...,
        description="One of: Expense Report, Requirement Document, Idea, Other"
    )
