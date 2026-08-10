from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app import models, schemas
from app.services.security import (
    get_password_hash,
    verify_password,
    create_access_token,
)

import requests
import re
import random
import smtplib
import os

from email.message import EmailMessage
from dotenv import load_dotenv


router = APIRouter()

load_dotenv()


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

SMTP_EMAIL = os.getenv("SMTP_EMAIL", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "").strip()


# ============================================================
# OTP EMAIL DELIVERY
# ============================================================

def send_otp_delivery(contact: str, otp: str) -> bool:
    """
    Send OTP to an email address.

    Priority:
        1. Resend API if RESEND_API_KEY exists
        2. Gmail SMTP
        3. Return False if both fail

    For phone numbers, SMS is not implemented yet.
    """

    print("\n" + "=" * 70)
    print("OTP DELIVERY STARTED")
    print(f"Recipient: {contact}")
    print(f"OTP generated: {otp}")
    print("=" * 70)

    # --------------------------------------------------------
    # Detect email
    # --------------------------------------------------------

    is_email = bool(
        re.match(
            r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
            contact
        )
    )

    if not is_email:

        print("📱 SMS OTP requested.")
        print("⚠️ SMS provider is not configured.")
        print(f"DEBUG OTP: {otp}")

        return False

    # ========================================================
    # METHOD 1: RESEND
    # ========================================================

    if RESEND_API_KEY:

        print("Trying Resend email delivery...")

        try:

            response = requests.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": "FinanceVision Security <onboarding@resend.dev>",
                    "to": [contact],
                    "subject": "FinanceVision - Security Verification Code",
                    "html": f"""
                    <!DOCTYPE html>
                    <html>
                    <body style="
                        margin:0;
                        padding:30px;
                        background:#0a0a0a;
                        font-family:Arial,sans-serif;
                        color:white;
                    ">

                        <div style="
                            max-width:600px;
                            margin:auto;
                            background:#111111;
                            padding:30px;
                            border-radius:12px;
                            border:1px solid #333333;
                        ">

                            <h2 style="
                                color:#ef4444;
                                margin-bottom:20px;
                            ">
                                FinanceVision
                            </h2>

                            <p>
                                Your 6-digit security verification code is:
                            </p>

                            <div style="
                                margin:30px 0;
                                padding:20px;
                                text-align:center;
                                background:#000000;
                                border:1px solid #dc2626;
                                border-radius:10px;
                            ">

                                <span style="
                                    font-size:36px;
                                    font-weight:bold;
                                    letter-spacing:8px;
                                    color:#ffffff;
                                ">
                                    {otp}
                                </span>

                            </div>

                            <p style="
                                color:#999999;
                                font-size:13px;
                            ">
                                If you did not request this code,
                                please ignore this email.
                            </p>

                            <p style="
                                color:#666666;
                                font-size:12px;
                                margin-top:30px;
                            ">
                                FinanceVision Security Team
                            </p>

                        </div>

                    </body>
                    </html>
                    """,
                },
                timeout=15,
            )

            print(f"Resend HTTP status: {response.status_code}")

            if response.status_code in (200, 201):

                print(
                    f"✅ OTP successfully sent via Resend to {contact}"
                )

                return True

            print(
                "❌ Resend failed."
            )

            print(
                f"Resend response: {response.text}"
            )

        except Exception as e:

            print("❌ Resend exception")
            print(f"Error type: {type(e).__name__}")
            print(f"Error: {e}")

    # ========================================================
    # METHOD 2: GMAIL SMTP
    # ========================================================

    print("\nTrying Gmail SMTP delivery...")

    if not SMTP_EMAIL:

        print("❌ SMTP_EMAIL environment variable is missing.")

    if not SMTP_PASSWORD:

        print("❌ SMTP_PASSWORD environment variable is missing.")

    if SMTP_EMAIL and SMTP_PASSWORD:

        try:

            msg = EmailMessage()

            msg["Subject"] = (
                "FinanceVision - "
                "Secure Identity Verification Code"
            )

            msg["From"] = SMTP_EMAIL
            msg["To"] = contact

            msg.set_content(
                f"""
Welcome to FinanceVision.

Your 6-digit security verification code is:

{otp}

Please enter this code in FinanceVision to verify your account.

If you did not request this code, please ignore this email.

FinanceVision Security Team
"""
            )

            print("Connecting to Gmail SMTP...")
            print("SMTP server: smtp.gmail.com")
            print("SMTP port: 465")

            with smtplib.SMTP_SSL(
                "smtp.gmail.com",
                465,
                timeout=20,
            ) as server:

                print("✅ SMTP connection established")

                server.login(
                    SMTP_EMAIL,
                    SMTP_PASSWORD,
                )

                print("✅ SMTP authentication successful")

                server.send_message(msg)

                print(
                    f"✅ OTP EMAIL SENT SUCCESSFULLY TO {contact}"
                )

            return True

        except smtplib.SMTPAuthenticationError as e:

            print("❌ Gmail SMTP authentication failed.")
            print(f"Error: {e}")

            print(
                "\nIMPORTANT:"
            )
            print(
                "SMTP_PASSWORD must be a Google App Password."
            )
            print(
                "Do NOT use your normal Gmail password."
            )

        except smtplib.SMTPException as e:

            print("❌ Gmail SMTP error.")
            print(f"Error type: {type(e).__name__}")
            print(f"Error: {e}")

        except Exception as e:

            print("❌ Gmail SMTP failed.")
            print(f"Error type: {type(e).__name__}")
            print(f"Error: {e}")

    # ========================================================
    # DELIVERY FAILED
    # ========================================================

    print("\n" + "=" * 70)
    print("❌ OTP EMAIL DELIVERY FAILED")
    print(f"Recipient: {contact}")
    print(f"DEBUG OTP: {otp}")
    print("=" * 70 + "\n")

    return False


# ============================================================
# SIGNUP
# ============================================================

@router.post("/signup")
def signup(
    user: schemas.UserCreate,
    db: Session = Depends(get_db),
):

    contact = user.contact_id.strip()

    is_email = bool(
        re.match(
            r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
            contact
        )
    )

    is_phone = (
        contact.isdigit()
        and len(contact) >= 10
    )

    if not is_email and not is_phone:

        raise HTTPException(
            status_code=400,
            detail="Must provide a valid Email or Mobile Number.",
        )

    # --------------------------------------------------------
    # Password validation
    # --------------------------------------------------------

    if (
        len(user.password) < 8
        or not re.search(r"[A-Z]", user.password)
        or not re.search(r"\d", user.password)
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Password must be 8+ chars "
                "with 1 uppercase letter and 1 number."
            ),
        )

    # --------------------------------------------------------
    # Generate OTP
    # --------------------------------------------------------

    otp = str(
        random.randint(
            100000,
            999999,
        )
    )

    hashed_pw = get_password_hash(
        user.password
    )

    # --------------------------------------------------------
    # Check existing user
    # --------------------------------------------------------

    existing_user = (
        db.query(models.User)
        .filter(
            models.User.contact_id == contact
        )
        .first()
    )

    if existing_user:

        if existing_user.is_verified:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Account already registered. "
                    "Please log in."
                ),
            )

        existing_user.otp_code = otp
        existing_user.hashed_password = hashed_pw

        db.commit()

    else:

        new_user = models.User(
            contact_id=contact,
            hashed_password=hashed_pw,
            is_verified=False,
            otp_code=otp,
        )

        db.add(new_user)
        db.commit()

    # --------------------------------------------------------
    # Send OTP
    # --------------------------------------------------------

    delivered = send_otp_delivery(
        contact,
        otp,
    )

    if not delivered:

        raise HTTPException(
            status_code=500,
            detail=(
                "OTP was generated but could not be "
                "delivered. Check server email configuration."
            ),
        )

    return {
        "message": (
            "Security code sent successfully. "
            "Please check your email."
        )
    }


# ============================================================
# RESEND OTP
# ============================================================

@router.post("/resend-otp")
def resend_otp(
    payload: dict,
    db: Session = Depends(get_db),
):

    contact = payload.get(
        "contact_id",
        "",
    ).strip()

    if not contact:

        raise HTTPException(
            status_code=400,
            detail="Contact ID is required.",
        )

    user = (
        db.query(models.User)
        .filter(
            models.User.contact_id == contact
        )
        .first()
    )

    if not user:

        raise HTTPException(
            status_code=404,
            detail="User account not found.",
        )

    # --------------------------------------------------------
    # Generate new OTP
    # --------------------------------------------------------

    otp = str(
        random.randint(
            100000,
            999999,
        )
    )

    user.otp_code = otp

    db.commit()

    delivered = send_otp_delivery(
        contact,
        otp,
    )

    if not delivered:

        raise HTTPException(
            status_code=500,
            detail=(
                "OTP was generated but could not be "
                "delivered."
            ),
        )

    return {
        "message": (
            "A fresh 6-digit security code "
            "has been sent to your email."
        )
    }


# ============================================================
# VERIFY OTP
# ============================================================

@router.post("/verify")
def verify_otp(
    payload: schemas.VerifyOTP,
    db: Session = Depends(get_db),
):

    user = (
        db.query(models.User)
        .filter(
            models.User.contact_id
            == payload.contact_id
        )
        .first()
    )

    if not user:

        raise HTTPException(
            status_code=400,
            detail="User account not found.",
        )

    if user.otp_code != payload.otp:

        raise HTTPException(
            status_code=400,
            detail="Invalid Verification Code.",
        )

    user.is_verified = True
    user.otp_code = None

    db.commit()

    access_token = create_access_token(
        data={
            "sub": user.contact_id
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


# ============================================================
# LOGIN
# ============================================================

@router.post(
    "/login",
    response_model=schemas.Token,
)
def login(
    user: schemas.UserCreate,
    db: Session = Depends(get_db),
):

    db_user = (
        db.query(models.User)
        .filter(
            models.User.contact_id
            == user.contact_id
        )
        .first()
    )

    if (
        not db_user
        or not verify_password(
            user.password,
            db_user.hashed_password,
        )
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid credentials.",
        )

    if not db_user.is_verified:

        raise HTTPException(
            status_code=403,
            detail=(
                "Account not verified. "
                "Please verify your OTP."
            ),
        )

    access_token = create_access_token(
        data={
            "sub": db_user.contact_id
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


# ============================================================
# GOOGLE AUTH
# ============================================================

@router.post(
    "/google",
    response_model=schemas.Token,
)
def google_auth(
    payload: dict,
    db: Session = Depends(get_db),
):

    token = payload.get("token")

    if not token:

        raise HTTPException(
            status_code=400,
            detail="No Google token provided.",
        )

    try:

        response = requests.get(
            "https://oauth2.googleapis.com/tokeninfo",
            params={
                "id_token": token
            },
            timeout=15,
        )

        if response.status_code != 200:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid Google authentication token."
                ),
            )

        user_info = response.json()

        email = user_info.get("email")

        if not email:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract email from Google."
                ),
            )

        user = (
            db.query(models.User)
            .filter(
                models.User.contact_id == email
            )
            .first()
        )

        if not user:

            user = models.User(
                contact_id=email,
                hashed_password="google_sso_no_password",
                is_verified=True,
            )

            db.add(user)
            db.commit()
            db.refresh(user)

        access_token = create_access_token(
            data={
                "sub": user.contact_id
            }
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Google Auth Error: {str(e)}",
        )