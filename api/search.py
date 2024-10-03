# app/api/search.py
from fastapi import APIRouter, File, UploadFile, Form, Depends
from config import load_models, initialize_pinecone
from utils import process_query, search_similar_documents
from schemas.models import SearchResponse, SearchResult

router = APIRouter()


@router.post("/search/", response_model=SearchResponse)
async def search(
    file: UploadFile = File(...),
    query_text: str = Form(...),
    models: tuple = Depends(load_models),
    index: callable = Depends(initialize_pinecone),
):
    model, processor, vision_projection, text_projection = models
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    query_embedding = process_query(
        file_path, query_text, model, processor, vision_projection, text_projection
    )
    raw_results = search_similar_documents(query_embedding, index)

    # Convert raw results to SearchResult objects
    results = [
        SearchResult(doc_id=match["id"], score=match["score"])
        for match in raw_results["matches"]
    ]

    return SearchResponse(results=results)
