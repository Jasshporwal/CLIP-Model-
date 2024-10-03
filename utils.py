from PIL import Image
import fitz
import torch
import io


def process_image(image, model, processor, vision_projection):
    inputs = processor(images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.vision_model(**inputs.to(model.device))
    image_features = outputs.pooler_output
    return vision_projection(image_features)


def process_document(image_path, doc_id, model, processor, vision_projection, index):
    try:
        if image_path.lower().endswith(".pdf"):
            pdf_document = fitz.open(image_path)
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                pix = page.get_pixmap()
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                projected_features = process_image(
                    image, model, processor, vision_projection
                )
                index.upsert(
                    [
                        (
                            f"{doc_id}_page_{page_num}",
                            projected_features.squeeze().tolist(),
                        )
                    ]
                )
        else:
            image = Image.open(image_path)
            projected_features = process_image(
                image, model, processor, vision_projection
            )
            index.upsert([(doc_id, projected_features.squeeze().tolist())])

        print(f"Successfully processed document: {image_path}")
        return doc_id
    except Exception as e:
        print(f"Error processing document {image_path}: {str(e)}")
        raise


def process_query(
    file_path, query_text, model, processor, vision_projection, text_projection
):
    # Check if the file is a PDF
    if file_path.lower().endswith(".pdf"):
        # Open the PDF and convert the first page to an image
        pdf_document = fitz.open(file_path)
        first_page = pdf_document[0]
        pix = first_page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    else:
        # If it's not a PDF, assume it's an image file
        image = Image.open(file_path)

    inputs = processor(
        text=query_text,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    vision_inputs = {
        k: v.to(model.device) for k, v in inputs.items() if k in ["pixel_values"]
    }
    text_inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
        if k in ["input_ids", "attention_mask"]
    }

    with torch.no_grad():
        vision_outputs = model.vision_model(**vision_inputs)
        text_outputs = model.text_model(**text_inputs)

    image_features = vision_outputs.pooler_output
    text_features = text_outputs.pooler_output

    projected_image = vision_projection(image_features)
    projected_text = text_projection(text_features)

    combined_query = (projected_image + projected_text) / 2
    return combined_query.squeeze().detach().numpy()


def search_similar_documents(query_embedding, index, top_k=5):
    results = index.query(
        vector=query_embedding.tolist(), top_k=top_k, include_metadata=True
    )
    return results
