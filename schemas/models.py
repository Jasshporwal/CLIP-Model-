
from pydantic import BaseModel
from typing import List
import torch


class DocumentUpload(BaseModel):
    doc_id: str
    filename: str
    status: str


class DocumentUploadResponse(BaseModel):
    uploaded_documents: List[DocumentUpload]


class SearchResult(BaseModel):
    doc_id: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


class ProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
