from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Import our routers and database setup
from app.api import chat, auth
from app.database import engine
from app import models

# CRITICAL: This tells SQLAlchemy to create tables if they do not exist
models.Base.metadata.create_all(bind=engine)

# Initialize FastAPI App (Rebranded to FinNova)
app = FastAPI(title="FinNova AI | Institutional Trading Engine")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include our API endpoints (THIS FIXES THE GOOGLE "NOT FOUND" ERROR)
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

# Host the index.html file directly from the server
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")