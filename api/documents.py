# # app/api/documents.py
# from fastapi import APIRouter, File, UploadFile, Form, Depends
# from config import load_models, initialize_pinecone
# from utils import process_document
# from models import Document

# router = APIRouter()

# @router.post("/upload_document/", response_model=Document)
# async def upload_document(
#     file: UploadFile = File(...),
#     doc_id: str = Form(...),
#     models: tuple = Depends(load_models),
#     index: callable = Depends(initialize_pinecone)
# ):
#     model, processor, vision_projection, _ = models
#     file_path = f"/tmp/{file.filename}"
#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     processed_doc_id = process_document(file_path, doc_id, model, processor, vision_projection, index)
#     return Document(doc_id=processed_doc_id, filename=file.filename)

# app/api/documents.py
from fastapi import APIRouter, File, UploadFile, Form, Depends
from typing import List
from config import load_models, initialize_pinecone
from utils import process_document
from schemas.models import DocumentUploadResponse, DocumentUpload

router = APIRouter()


@router.post("/upload_documents/", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    doc_ids: List[str] = Form(...),
    models: tuple = Depends(load_models),
    index: callable = Depends(initialize_pinecone),
):
    print(f"Received {len(files)} files and {len(doc_ids)} document IDs")
    print(f"doc_ids: {doc_ids}")
    print(f"files: {files}, type: {type(files)}")
    if len(files) != len(doc_ids):
        raise ValueError("Number of files must match number of document IDs")

    model, processor, vision_projection, _ = models
    uploaded_documents = []

    for file, doc_id in zip(files, doc_ids):
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        try:
            processed_doc_id = process_document(
                file_path, doc_id, model, processor, vision_projection, index
            )
            status = "success" if processed_doc_id else "failed"
        except Exception as e:
            status = f"error: {str(e)}"

        uploaded_documents.append(
            DocumentUpload(doc_id=doc_id, filename=file.filename, status=status)
        )

    return DocumentUploadResponse(uploaded_documents=uploaded_documents)
