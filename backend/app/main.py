from fastapi import FastAPI
from .api import router

app = FastAPI(title="Soccer Game Analysis API")

app.include_router(router)
