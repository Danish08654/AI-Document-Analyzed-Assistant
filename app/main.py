from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="AI Business OS"
)

app.include_router(router)