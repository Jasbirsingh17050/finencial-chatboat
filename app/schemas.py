from pydantic import BaseModel

class UserCreate(BaseModel):
    # UPGRADE: This now accepts an email OR a mobile number
    contact_id: str 
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class VerifyOTP(BaseModel):
    contact_id: str
    otp: str