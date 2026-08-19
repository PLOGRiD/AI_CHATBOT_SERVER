from fastapi import FastAPI

from api.routes import router as chat_router
from api.routes import waste_router

app = FastAPI(title="plogrid-chat")

app.include_router(chat_router)
app.include_router(waste_router)

@app.get("/")
def read_root():
    return {"message": "plogrid"}