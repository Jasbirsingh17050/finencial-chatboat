from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. This is the file where our database will be saved locally
SQLALCHEMY_DATABASE_URL = "sqlite:///./quantfi.db"

# 2. We create an 'engine' to talk to the SQLite file
# 'check_same_thread' is needed for SQLite in FastAPI to prevent crashes
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 3. This is the factory that creates database 'sessions' for our users
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. This Base class is what our database tables will inherit from
Base = declarative_base()

# 5. A helper function we will use later to get a database session and close it safely
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()