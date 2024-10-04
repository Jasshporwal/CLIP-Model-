from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from api import documents, search
import json


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return jsonable_encoder(obj)
        except:
            return str(obj)


app = FastAPI()
app.json_encoder = CustomJSONEncoder

app.include_router(documents.router)
app.include_router(search.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
