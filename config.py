import os
from pinecone import Pinecone, ServerlessSpec
from transformers import CLIPProcessor, CLIPModel
from schemas.models import ProjectionLayer


def initialize_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Pinecone API key not found in environment variables")

    pc = Pinecone(api_key=api_key)
    index_name = os.getenv("PINECONE_INDEX_NAME", "default_index")
    dimension = 256

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists. Using the existing index.")

    return pc.Index(index_name)


def load_models():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    vision_input_dim = model.vision_model.config.hidden_size
    text_input_dim = model.text_model.config.hidden_size
    output_dim = 256

    vision_projection = ProjectionLayer(vision_input_dim, output_dim)
    text_projection = ProjectionLayer(text_input_dim, output_dim)

    return model, processor, vision_projection, text_projection
