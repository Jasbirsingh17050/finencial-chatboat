from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app import models, schemas
from app.services.security import get_password_hash, verify_password, create_access_token
import requests
import re
import random
import smtplib
import os
from email.message import EmailMessage
from dotenv import load_dotenv

router = APIRouter()

load_dotenv()
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "jasbirsingh17050@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "Sahilthakur5940@")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")

def send_otp_delivery(contact: str, otp: str) -> bool:
    """Delivers 6-digit OTP via Resend HTTP API first, then SMTP SSL, then mock logs."""
    is_email = "@" in contact and "." in contact
    if not is_email:
        print("\n" + "="*50)
        print(f"📱 SMS OTP MOCK DELIVERED TO: {contact}")
        print(f"🔑 SECRET OTP: {otp}")
        print("="*50 + "\n")
        return True

    if RESEND_API_KEY:
        try:
            res = requests.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "from": "FinanceVision Security <onboarding@resend.dev>",
                    "to": [contact],
                    "subject": "FinanceVision - 6-Digit Identity Verification Code",
                    "html": f"""
                    <div style="font-family: Arial, sans-serif; background-color: #0a0a0a; color: #ffffff; padding: 30px; border-radius: 12px;">
                        <h2 style="color: #ef4444;">FinanceVision Terminal Access</h2>
                        <p style="color: #94a3b8;">Your 6-digit identity verification code is:</p>
                        <h1 style="font-size: 36px; letter-spacing: 6px; color: #ffffff; background: #111; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #dc2626;">{otp}</h1>
                        <p style="color: #64748b; font-size: 12px;">If you did not request this code, please ignore this message.</p>
                    </div>
                    """
                },
                timeout=5
            )
            if res.status_code in [200, 201]:
                print(f"✅ OTP successfully sent via Resend API to {contact}")
                return True
        except Exception as e:
            print(f"⚠️ Resend API delivery failed: {e}")

    if SMTP_EMAIL and SMTP_PASSWORD:
        try:
            msg = EmailMessage()
            msg.set_content(f"Welcome to FinanceVision!\n\nYour 6-digit security code is: {otp}\n\nEnter this in the terminal to verify your clearance.")
            msg['Subject'] = "FinanceVision - Secure Identity Verification Code"
            msg['From'] = SMTP_EMAIL
            msg['To'] = contact
            
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=5)
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f"✅ OTP successfully sent via SMTP SSL to {contact}")
            return True
        except Exception as e:
            print(f"⚠️ SMTP SSL failed: {e}")

    print("\n" + "="*50)
    print(f"⚠️ SMTP/API UNREACHABLE. MOCKING OTP SEND...")
    print(f"📧 RECIPIENT: {contact}")
    print(f"🔑 YOUR SECRET OTP CODE IS: {otp}")
    print("="*50 + "\n")
    return False

@router.post("/signup")
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    contact = user.contact_id.strip()
    is_email = "@" in contact and "." in contact
    is_phone = contact.isdigit() and len(contact) >= 10
    
    if not is_email and not is_phone:
        raise HTTPException(status_code=400, detail="Must provide a valid Email or Mobile Number.")

    if len(user.password) < 8 or not re.search(r"[A-Z]", user.password) or not re.search(r"\d", user.password):
        raise HTTPException(status_code=400, detail="Password must be 8+ chars with 1 uppercase letter and 1 number.")

    existing_user = db.query(models.User).filter(models.User.contact_id == contact).first()
    otp = str(random.randint(100000, 999999))
    hashed_pw = get_password_hash(user.password)

    if existing_user:
        if existing_user.is_verified:
            raise HTTPException(status_code=400, detail="Account already registered. Please log in.")
        else:
            existing_user.otp_code = otp
            existing_user.hashed_password = hashed_pw
            db.commit()
    else:
        new_user = models.User(contact_id=contact, hashed_password=hashed_pw, is_verified=False, otp_code=otp)
        db.add(new_user)
        db.commit()
    
    send_otp_delivery(contact, otp)
    return {"message": "Security code deployed successfully!"}

@router.post("/resend-otp")
def resend_otp(payload: dict, db: Session = Depends(get_db)):
    contact = payload.get("contact_id", "").strip()
    if not contact:
        raise HTTPException(status_code=400, detail="Contact ID is required.")
    
    user = db.query(models.User).filter(models.User.contact_id == contact).first()
    if not user:
        raise HTTPException(status_code=404, detail="User account not found.")
    
    otp = str(random.randint(100000, 999999))
    user.otp_code = otp
    db.commit()
    
    send_otp_delivery(contact, otp)
    return {"message": "A fresh 6-digit security code has been deployed."}

@router.post("/verify")
def verify_otp(payload: schemas.VerifyOTP, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.contact_id == payload.contact_id).first()
    if not user or user.otp_code != payload.otp:
        raise HTTPException(status_code=400, detail="Invalid Verification Code.")
    
    user.is_verified = True
    user.otp_code = None 
    db.commit()

    access_token = create_access_token(data={"sub": user.contact_id})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login", response_model=schemas.Token)
def login(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.contact_id == user.contact_id).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    
    if not db_user.is_verified:
        raise HTTPException(status_code=403, detail="Account not verified. Deploying new security code.")
    
    access_token = create_access_token(data={"sub": db_user.contact_id})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/google", response_model=schemas.Token)
def google_auth(payload: dict, db: Session = Depends(get_db)):
    token = payload.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="No Google token provided.")
    try:
        response = requests.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={token}")
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Invalid Google authentication token.")
        user_info = response.json()
        email = user_info.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Could not extract email from Google.")
            
        user = db.query(models.User).filter(models.User.contact_id == email).first()
        if not user:
            user = models.User(contact_id=email, hashed_password="google_sso_no_password", is_verified=True)
            db.add(user)
            db.commit()
            db.refresh(user)
            
        access_token = create_access_token(data={"sub": user.contact_id})
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Auth Error: {str(e)}")