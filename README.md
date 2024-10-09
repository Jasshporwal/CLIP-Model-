
# CLIP API

The CLIP application is a document search and upload platform built using FastAPI, Pydantic, and Pinecone. It allows users to upload documents, process them, and search for similar documents using a query.

## Features

* Document Upload: Users can upload documents in various formats, including PDF and images.
* Document Processing: The application processes uploaded documents using a machine learning model to extract features.
* Search: Users can search for similar documents using a query, and the application returns a list of relevant documents.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/Jasshporwal/CLIP-Model-.git
   cd CLIP-MODEL-
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   uvicorn main:app --reload
   ```

The server will start running on `http://localhost:8000`.


## Configuration

The application uses environment variables to configure the Pinecone API key and index name. Set the following environment variables:
```
PINECONE_API_KEY=your_api_key
PINECONE_INDEX_NAME=your_index_name
```
## Running the Application

To run the application, execute the following command:
```
uvicorn main:app --reload
```
The server will start running on `http://localhost:8000`.

## API Endpoints

The application provides the following API endpoints:

* `/upload_documents`: Upload multiple documents
* `/search`: Search for similar documents using a query


## Contributing

Contributions are welcome! Please submit a pull request with your changes.

