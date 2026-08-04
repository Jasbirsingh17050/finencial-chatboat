import bcrypt
import jwt
from datetime import datetime, timedelta

SECRET_KEY = "quantfi_super_secret_master_key_12345"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # Token lasts for 7 days

def get_password_hash(password: str) -> str:
    """Scrambles the password using pure, modern bcrypt."""
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password=pwd_bytes, salt=salt)
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Checks if the typed password matches the scrambled one in the database."""
    password_byte_enc = plain_password.encode('utf-8')
    hashed_password_byte_enc = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_byte_enc, hashed_password_byte_enc)

def create_access_token(data: dict) -> str:
    """Creates the JWT digital wristband for the user."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    # FIX: Prevent Token Serialization Bug (Bytes vs String issue)
    if isinstance(encoded_jwt, bytes):
        return encoded_jwt.decode('utf-8')
        
    return encoded_jwt