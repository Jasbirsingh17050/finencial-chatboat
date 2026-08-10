import os
import yfinance as yf
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
import jwt

from app.services.llm import stream_financial_response, process_uploaded_file, get_market_overview, get_global_indices
from app.database import get_db
from app import models
from app.services.security import SECRET_KEY, ALGORITHM

router = APIRouter()

async def get_current_user(authorization: str = Header(None), db: Session = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid wristband")
    
    token = authorization.split(" ")[1]
    
    # FIX: Clean up the token just in case the browser stored it poorly
    if token.startswith("b'") and token.endswith("'"):
        token = token[2:-1]
    if token in ["null", "undefined", ""]:
        raise HTTPException(status_code=401, detail="Token is missing")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        contact_id: str = payload.get("sub")
        
        user = db.query(models.User).filter(models.User.contact_id == contact_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except Exception as e:
        # Bold warning in the terminal so we can see exactly why it failed!
        print("\n" + "="*50)
        print(f"⚠️ SECURITY TOKEN REJECTED ⚠️")
        print(f"Reason: {str(e)}")
        print(f"Token Value: {token}")
        print("="*50 + "\n")
        raise HTTPException(status_code=401, detail="Expired or fake wristband")

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    result = process_uploaded_file(file_path, file.filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    if result == "Success":
        return JSONResponse(content={"message": f"Successfully learned from {file.filename}!"})
    else:
        return JSONResponse(content={"message": result}, status_code=400)

@router.get("/market")
async def get_market_data():
    return JSONResponse(content=get_market_overview())

@router.get("/indices")
async def get_indices_data():
    return JSONResponse(content=get_global_indices())

# --- UPGRADED: USER PROFILE ENDPOINTS (RAG FOCUS) ---
@router.get("/profile")
async def get_profile(user: models.User = Depends(get_current_user)):
    return {
        "contact_id": user.contact_id,
        "full_name": user.full_name,
        "gender": user.gender,
        "professional_role": user.professional_role,
        "focus_area": user.focus_area,
        "profile_picture": user.profile_picture
    }

@router.put("/profile")
async def update_profile(payload: dict, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    user.full_name = payload.get("full_name", user.full_name)
    user.gender = payload.get("gender", user.gender)
    user.professional_role = payload.get("professional_role", user.professional_role)
    user.focus_area = payload.get("focus_area", user.focus_area)
    db.commit()
    return {"message": "Profile updated"}

@router.post("/profile/picture")
async def upload_profile_picture(file: UploadFile = File(...), user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    os.makedirs("uploads/profiles", exist_ok=True)
    import time
    file_path = f"uploads/profiles/{user.id}_{int(time.time())}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    user.profile_picture = f"/{file_path}"
    db.commit()
    return {"profile_picture": user.profile_picture}

@router.get("/chart_data/{ticker}")
async def get_chart_data(ticker: str, period: str = "1mo", user: models.User = Depends(get_current_user)):
    try:
        stock = yf.Ticker(ticker)
        interval_map = {"1d": "5m", "5d": "15m", "1mo": "1d", "6mo": "1d", "ytd": "1d", "1y": "1d", "5y": "1wk", "max": "1mo"}
        hist = stock.history(period=period, interval=interval_map.get(period, "1d"))
        if hist.empty: return JSONResponse(content={"labels": [], "prices": []})
        hist = hist.dropna(subset=['Close'])
        labels = hist.index.strftime('%I:%M %p' if period in ["1d", "5d"] else '%b %Y' if period in ["max", "5y"] else '%b %d, %Y').tolist()
        return JSONResponse(content={"labels": labels, "prices": hist['Close'].round(2).tolist()})
    except Exception as e:
        return JSONResponse(content={"labels": [], "prices": []}, status_code=400)

@router.get("/sessions")
async def get_sessions(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(models.ChatSession).filter(models.ChatSession.user_id == user.id).order_by(models.ChatSession.created_at.desc()).all()
    return [{"id": s.id, "title": s.title} for s in sessions]

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(models.ChatSession).filter(models.ChatSession.id == session_id, models.ChatSession.user_id == user.id).first()
    if not session: raise HTTPException(status_code=404, detail="Session not found")
    db.query(models.ChatMessage).filter(models.ChatMessage.session_id == session_id).delete()
    db.delete(session)
    db.commit()
    return {"message": "Session deleted successfully"}

@router.get("/history/{session_id}")
async def get_history(session_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(models.ChatSession).filter(models.ChatSession.id == session_id, models.ChatSession.user_id == user.id).first()
    if not session: return []
    messages = db.query(models.ChatMessage).filter(models.ChatMessage.session_id == session_id).order_by(models.ChatMessage.created_at).all()
    return [{"sender": msg.sender, "text": msg.text, "timestamp": msg.created_at.isoformat() + "Z"} for msg in messages]

@router.post("/stream")
async def chat_stream(payload: dict, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    query = payload.get("query", "")
    session_id = payload.get("session_id")
    
    if session_id:
        chat_session = db.query(models.ChatSession).filter(models.ChatSession.id == session_id, models.ChatSession.user_id == user.id).first()
    else:
        title = query[:25] + "..." if len(query) > 25 else query
        chat_session = models.ChatSession(user_id=user.id, title=title)
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)

    current_session_id = chat_session.id

    # Feed the new professional profile to the AI
    user_profile = {
        "full_name": user.full_name,
        "professional_role": user.professional_role,
        "focus_area": user.focus_area
    }

    async def response_generator():
        full_response = ""
        try:
            for chunk in stream_financial_response(query, user_profile):
                full_response += chunk
                yield chunk
        except Exception as e:
            print(f"Stream interrupted: {e}")
        finally:
            from app.database import SessionLocal 
            save_db = SessionLocal()
            try:
                save_db.add(models.ChatMessage(sender="User", text=query, session_id=current_session_id))
                save_db.add(models.ChatMessage(sender="Bot", text=full_response, session_id=current_session_id))
                save_db.commit()
            except Exception as e:
                print(f"Error saving messages: {e}")
            finally:
                save_db.close()

    return StreamingResponse(
        response_generator(), 
        media_type="text/event-stream",
        headers={"X-Session-ID": str(current_session_id)}
    )