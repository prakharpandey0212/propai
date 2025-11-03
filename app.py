# --- INSTALLATION INSTRUCTIONS (These are comments, not code) ---
# 1. Activate Virtual Environment (venv)
# 2. pip install fastapi uvicorn pydantic google-genai sqlalchemy databases bcrypt aiosqlite python-dotenv python-jose
#3. Server run karne ke liye: uvicorn app:app --reload
# -----------------------------------------------------------------

from fastapi import FastAPI, HTTPException, Depends, status, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
import os
import json
import asyncio 
from contextlib import asynccontextmanager 
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv
from jose import jwt, JWTError

from databases import Database
from sqlalchemy import create_engine, Column, Integer, String, Boolean, MetaData, Table, DateTime
import datetime
import bcrypt
import time 

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY") 
DUMMY_KEY = "AIzaSyCX4XNnhSmFbjK3dUv4B_1dd5qBcVIBds8"
SECRET_KEY = os.environ.get("SECRET_KEY", "your-super-secret-key-replace-this") 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 Days

if not API_KEY or API_KEY == DUMMY_KEY:
    print("CRITICAL: GEMINI_API_KEY is missing or using placeholder. Analysis will fall back.")
    API_KEY = DUMMY_KEY

GEMINI_CLIENT = None
DAILY_CREDIT_LIMIT = 20

DATABASE_URL = "sqlite:///./prophai.db" 
database = Database(DATABASE_URL)
metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String(120), unique=True, index=True),
    Column("hashed_password", String(128)),
    Column("is_active", Boolean, default=True),
    Column("analysis_credits", Integer, default=DAILY_CREDIT_LIMIT), 
    Column("created_at", DateTime, default=datetime.datetime.utcnow),
)

analysis_results = Table(
    "analysis_results",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer), 
    Column("idea_summary", String(500)),
    Column("score", Integer),
    Column("created_at", DateTime, default=datetime.datetime.utcnow),
)

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def hash_password(password: str) -> str:
    """Hashes the password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies the plain password against the hashed password."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict):
    """Generates a new JWT token containing the user's ID."""
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    
    if 'sub' in to_encode:
        to_encode['sub'] = str(to_encode['sub'])
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def refresh_daily_credits():
    """Resets analysis credits to DAILY_CREDIT_LIMIT for all users."""
    print("Running daily credit refresh check...")
    update_query = users.update().values(analysis_credits=DAILY_CREDIT_LIMIT)
    await database.execute(update_query)
    print(f"Successfully reset credits to {DAILY_CREDIT_LIMIT} for all users.")


async def credit_refresh_scheduler():
    """Runs the credit refresh task every 24 hours."""
    REFRESH_INTERVAL = 86400
    await asyncio.sleep(5) 
    await refresh_daily_credits() 

    while True:
        try:
            await asyncio.sleep(REFRESH_INTERVAL)
            await refresh_daily_credits()
        except asyncio.CancelledError:
            print("Credit refresh scheduler cancelled.")
            break
        except Exception as e:
            print(f"Credit refresh task failed: {e}")
            await asyncio.sleep(60) 


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events (DB connection, Gemini client init, background task)."""
    global GEMINI_CLIENT
    
    if SECRET_KEY == "your-super-secret-key-replace-this":
        print("CRITICAL WARNING: SECRET_KEY is not set in .env. JWT is insecure!")
        
    metadata.create_all(engine)
    await database.connect()
    print("Database connected and schema checked.")

    refresh_task = asyncio.create_task(credit_refresh_scheduler())
    print("Daily credit refresh scheduler started.")

    try:
        if API_KEY != DUMMY_KEY:
            GEMINI_CLIENT = genai.Client(api_key=API_KEY)
        else:
            print("WARNING: Gemini Client skipped. Using fallback results.")
    except Exception as e:
        print(f"Error initializing Gemini Client: {e}")

    yield

    refresh_task.cancel()
    try:
        await refresh_task 
    except asyncio.CancelledError:
        pass

    await database.disconnect()
    print("Database connection closed.")
    
app = FastAPI(title="ProphAI Backend", lifespan=lifespan)

origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



async def get_current_user(token: str = Security(oauth2_scheme)):
    """Decodes JWT and fetches the current active user from the database."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub") 
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials (No user ID in token)",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user_id_int = int(user_id) 
    except (JWTError, ValueError) as e:
        print(f"JWT Decoding/Conversion Error: {e}") 
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials (Token expired or tampered)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    query = users.select().where(users.c.id == user_id_int)
    user = await database.fetch_one(query)

    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    
    return {
        "id": user['id'],
        "email": user['email'],
        "analysis_credits": user['analysis_credits'] if user['analysis_credits'] is not None else DAILY_CREDIT_LIMIT, 
    }


class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    credits: int

class UserIn(BaseModel):
    email: EmailStr
    password: str
class UserOut(BaseModel):
    id: int
    email: EmailStr
    analysis_credits: int
    created_at: datetime.datetime
class Login(BaseModel):
    email: EmailStr
    password: str
class IdeaRequest(BaseModel):
    idea: str
class TechStackDetail(BaseModel):
    component: str
    technology: str
    justification: str
class CompetitorDetail(BaseModel):
    competitor: str
    similarFeatures: str
    differentiation: str
class ChatRequest(BaseModel):
    message: str
class ChatResponse(BaseModel):
    reply: str
class AnalysisResult(BaseModel):
    noveltyScore: int
    trendsScore: int
    viralityPotential: int
    netWorthEstimate: str
    uniquenessSummary: str
    longTermViability: str
    ecosystemRisks: str
    adoptionDrivers: str
    actionableNextSteps: list[str]
    keywords: list[str]
    techAppeal: str
    techStack: list[TechStackDetail]
    projectPhases: list[dict] 
    competitiveLandscape: list[CompetitorDetail]

def create_analysis_prompt(idea: str) -> str:
    return f"""
    You are ProphAI, the Strategic Innovation Core. Your primary task is to **first validate** whether the following input describes a software project, product, or clear technological innovation.
    Input Idea: "{idea}"
    **--- CRITICAL INSTRUCTION ---**
    **CONDITION A (Invalid/Non-Project):** If the input is a general question (e.g., "What is AI?"), a single irrelevant word, or does NOT describe a product/innovation, you MUST return a strict JSON response where **noveltyScore is 0** and ALL scores (trendsScore, viralityPotential) are **0** (zero).
    * Use this specific error summary for general questions/irrelevant input: **"Please input a valid project or innovation in proper manner."**
    * Use this specific error summary if the input is vague but might pass the length check: **"No project described below."**
    * For Condition A, fill all descriptive fields (e.g., netWorthEstimate, viability) with **"N/A: Input Validation Failed"** and use placeholder lists with error messages.
    **CONDITION B (Valid Project):** If the input clearly describes a software project or innovation, proceed with the full professional analysis and provide realistic scores (0-100) and detailed descriptions.
    **REQUIRED JSON STRUCTURE (Strictly adhere to this format for both CONDITIONS A and B):**
    
    {{
        "noveltyScore": [Integer 0-100],
        "trendsScore": [Integer 0-100],
        "viralityPotential": [Integer 0-100],
        "netWorthEstimate": "String describing the 3-year valuation range (e.g., '₹5 Crore - ₹10 Crore (High Tech Value)' OR 'ZERO - Input Not Validated')",
        "uniquenessSummary": "A concise summary of the core product's USP, OR the specific error message from Condition A.",
        "longTermViability": "A concise assessment of the idea's resilience, OR 'N/A: Invalid Input'.",
        "ecosystemRisks": "Concise summary of 2-3 risks, OR 'N/A: Invalid Input'.",
        "adoptionDrivers": "Concise summary of 2-3 drivers, OR 'N/A: Invalid Input'.",
        "actionableNextSteps": [
            "1. (Practical Step or Error Message from Condition A)",
            "2. (Practical Step or Error Message from Condition A)",
            "3. (Practical Step or Error Message from Condition A)"
        ],
        "keywords": [
            "Validation_Error", 
            "Invalid_Input"
        ],
        "techAppeal": "A concise summary of the project's appeal, OR 'N/A: Invalid Input'.",
        "techStack": [
            {{"component": "Frontend/UI", "technology": "React/Next.js", "justification": "Modern, scalable component-based architecture."}},
            {{"component": "Backend/API", "technology": "FastAPI/GoLang", "justification": "High performance and asynchronous handling for real-time data."}},
            {{"component": "Database/Data Layer", "technology": "PostgreSQL/VectorDB", "justification": "Structured data and efficient vector search for AI embeddings."}}
        ],
        "projectPhases": [
            {{"phase": "Phase 1: Proof of Concept", "duration": "4 Weeks", "focus": "Core Logic & Data Integrity, Minimal UI."}},
            {{"phase": "Phase 2: Alpha Launch", "duration": "8 Weeks", "focus": "Complete MVP, Security Audit, Internal Testing."}}
        ],
        "competitiveLandscape": [
            {{"competitor": "Major Competitor A", "similarFeatures": "AI-driven content generation.", "differentiation": "Your niche focus."}},
            {{"competitor": "Niche Startup B", "similarFeatures": "Simple UI.", "differentiation": "Superior security."}}
        ]
    }}
    """


@app.get("/")
def read_root():
    return {"status": "ProphAI Backend is Running", "endpoints": ["/analyze", "/chat", "/auth", "/history"], "version": "1.0"}


@app.post("/auth/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def signup(user: UserIn):
    query = users.select().where(users.c.email == user.email)
    existing_user = await database.fetch_one(query)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered.")

    hashed_password = await asyncio.to_thread(hash_password, user.password)
    
    query = users.insert().values(email=user.email, hashed_password=hashed_password, analysis_credits=DAILY_CREDIT_LIMIT)
    last_record_id = await database.execute(query)

    new_user_query = users.select().where(users.c.id == last_record_id)
    new_user = await database.fetch_one(new_user_query)
    
    return dict(new_user)


@app.post("/auth/login", response_model=Token)
async def login_for_access_token(credentials: Login):
    query = users.select().where(users.c.email == credentials.email)
    user = await database.fetch_one(query)

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password.", headers={"WWW-Authenticate": "Bearer"})
    
    is_valid = await asyncio.to_thread(verify_password, credentials.password, user.hashed_password)

    if not is_valid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password.", headers={"WWW-Authenticate": "Bearer"})
    
    access_token = create_access_token(data={"sub": user.id}) 
    
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "user_id": user.id, 
        "credits": user.analysis_credits
    }


@app.post("/analyze", response_model=AnalysisResult) 
async def analyze_project(request: IdeaRequest, current_user: dict = Depends(get_current_user)):
    
    if current_user['analysis_credits'] <= 0:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"No credits remaining. Current credits: {current_user['analysis_credits']}. Please buy more credits."
        )
        
    default_data = {
        "noveltyScore": 0, "trendsScore": 0, "viralityPotential": 0,
        "netWorthEstimate": "ZERO - System Fallback", 
        "uniquenessSummary": "Analysis core disconnected.",
        "longTermViability": "N/A: Core failure.",
        "ecosystemRisks": "Check API key and network connection.",
        "adoptionDrivers": "Try running the server again.",
        "actionableNextSteps": ["1. Check logs."],
        "keywords": ["Fallback", "Error"],
        "techAppeal": "N/A",
        "techStack": [{"component": "System", "technology": "N/A", "justification": "Failure to connect."}],
        "projectPhases": [{"phase": "Error Phase", "duration": "N/A", "focus": "Troubleshoot."}],
        "competitiveLandscape": [{"competitor": "System Fallback", "similarFeatures": "N/A", "differentiation": "Connection Error"}] 
    }

    try:
        client = GEMINI_CLIENT
        if GEMINI_CLIENT is None:
            raise Exception("Gemini Client not initialized (Missing API Key)")

        prompt = create_analysis_prompt(request.idea)

        response = await asyncio.to_thread(
            client.models.generate_content,
            model='gemini-2.5-flash',
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        
        analysis_data = json.loads(response.text)
        
        if isinstance(analysis_data.get('ecosystemRisks'), list):
            analysis_data['ecosystemRisks'] = ' '.join(analysis_data['ecosystemRisks'])
        if isinstance(analysis_data.get('adoptionDrivers'), list):
            analysis_data['adoptionDrivers'] = ' '.join(analysis_data['adoptionDrivers'])
            
        final_result = AnalysisResult(**analysis_data)
        
    except Exception as e:
        print(f"Analysis/API Error: {e}")
        final_result = AnalysisResult(**default_data)

    if final_result.noveltyScore > 0 or final_result.trendsScore > 0 or final_result.viralityPotential > 0:
        new_credits = current_user['analysis_credits'] - 1
        update_query = users.update().where(users.c.id == current_user['id']).values(analysis_credits=new_credits)
        await database.execute(update_query)

        total_score = round((final_result.noveltyScore + final_result.trendsScore + final_result.viralityPotential) / 3)
        history_query = analysis_results.insert().values(
            user_id=current_user['id'], 
            idea_summary=request.idea[:450] + '...' if len(request.idea) > 450 else request.idea,
            score=total_score
        )
        await database.execute(history_query)
        
    return final_result


@app.get("/history", status_code=status.HTTP_200_OK)
async def get_user_history(current_user: dict = Depends(get_current_user)):
    """Fetches analysis history for the current user based on JWT."""
    
    user_id = current_user['id']
    
    query = analysis_results.select().where(analysis_results.c.user_id == user_id).order_by(analysis_results.c.created_at.desc()).limit(10)
    history = await database.fetch_all(query)
    
    formatted_history = [
        {
            "date": item['created_at'].strftime("%Y-%m-%d %I:%M %p") if item['created_at'] else 'N/A', 
            "idea": item['idea_summary'],
            "score": item['score']
        }
        for item in history
    ]
    
    return {"history": formatted_history}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    chat_prompt = f"""
    You are a friendly and knowledgeable technical assistant. Answer the user's question concisely. 
    If the question is about a project or innovation analysis, gently redirect them to use the main ProphAI form.
    User Question: {request.message}
    """
    try:
        client = GEMINI_CLIENT
        
        if GEMINI_CLIENT is None:
            raise Exception("Gemini Client not initialized (Missing API Key)")

        response = await asyncio.to_thread(
            client.models.generate_content,
            model='gemini-2.5-flash',
            contents=chat_prompt,
        )
        
        return ChatResponse(reply=response.text)

    except Exception as e:
        print(f"Chat Error: {e}")
        return ChatResponse(reply="Sorry, I can't connect to the chat service right now. (Missing API Key?)")