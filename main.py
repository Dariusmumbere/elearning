from fastapi import FastAPI, Depends, HTTPException, status, Form, UploadFile, File, BackgroundTasks, Request, Query, Header, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON, func, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime, timedelta
from typing import Optional, List, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
import shutil
import uuid
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, Response, RedirectResponse
import boto3
from botocore.exceptions import ClientError
import io
import requests
from urllib.parse import quote
import logging
import google.generativeai as genai
import json
import alembic
from alembic import command
from alembic.config import Config
import pdfkit
import tempfile
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.graphics.shapes import Drawing, Rect, Circle, Line, Polygon, String as GString
from reportlab.graphics import renderPDF
from reportlab.platypus import Flowable
import base64
from io import BytesIO
import hashlib
import asyncio
import math
import redis
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backblaze B2 Configuration - PRIVATE BUCKET
B2_BUCKET_NAME = "uploads-dir"
B2_ENDPOINT_URL = "https://s3.us-east-005.backblazeb2.com"
B2_KEY_ID = os.getenv("B2_KEY_ID", "0055ca7845641d30000000002")
B2_APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY", "K005NNeGM9r28ujQ3jvNEQy2zUiu0TI")

# Initialize B2 client
b2_client = boto3.client(
    's3',
    endpoint_url=B2_ENDPOINT_URL,
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APPLICATION_KEY
)

# ---------------------------------------------------------------------------
# PesaPal Configuration
# ---------------------------------------------------------------------------
PESAPAL_CONSUMER_KEY = os.getenv("PESAPAL_CONSUMER_KEY", "lGw3V7l9BwOqZKttLM3Z8KcmopU1+tT1")
PESAPAL_CONSUMER_SECRET = os.getenv("PESAPAL_CONSUMER_SECRET", "hY5oqA0JGl4MwRCYFjn0y5n9xEs=")
# Switch to https://pay.pesapal.com/v3 for production
PESAPAL_BASE_URL = os.getenv("PESAPAL_BASE_URL", "https://pay.pesapal.com/v3")
PESAPAL_IPN_URL = os.getenv("PESAPAL_IPN_URL", "https://elearning-1-r5di.onrender.com/payments/ipn")
PESAPAL_CALLBACK_URL = os.getenv("PESAPAL_CALLBACK_URL", "https://online-coderise.vercel.app/payment/callback")
COURSE_PRICE_UGX = 25000  # UGX 25,000 per month per course

# In-memory cache for PesaPal token and IPN ID (survives process restarts via env fallback)
_pesapal_token_cache: dict = {}
_pesapal_ipn_id: Optional[str] = None

# Redis configuration for caching quiz questions
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    redis_client.ping()
    logger.info("Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
    redis_client = None

# In-memory cache fallback
quiz_cache = {}

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_cjvA0Qyglx7P@ep-winter-cloud-app1qkm3.c-7.us-east-1.aws.neon.tech/neondb?sslmode=require")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------------------------------------------------------------------
# Database Models
# ---------------------------------------------------------------------------

class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_instructor = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    # --- NEW profile fields ---
    bio = Column(Text, nullable=True)
    profile_image_filename = Column(String, nullable=True)
    location = Column(String, nullable=True)
    website = Column(String, nullable=True)
    phone = Column(String, nullable=True)

    courses = relationship("CourseModel", back_populates="instructor")
    enrollments = relationship("EnrollmentModel", back_populates="user")
    quiz_attempts = relationship("QuizAttemptModel", back_populates="user")
    certificates = relationship("CertificateModel", back_populates="user")
    payments = relationship("PaymentModel", back_populates="user")


class CourseModel(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    instructor_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_published = Column(Boolean, default=False)
    image_url = Column(String, nullable=True)
    image_filename = Column(String, nullable=True)

    instructor = relationship("UserModel", back_populates="courses")
    modules = relationship("ModuleModel", back_populates="course")
    enrollments = relationship("EnrollmentModel", back_populates="course")
    certificates = relationship("CertificateModel", back_populates="course")
    payments = relationship("PaymentModel", back_populates="course")


class ModuleModel(Base):
    __tablename__ = "modules"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    course_id = Column(Integer, ForeignKey("courses.id"))
    order = Column(Integer)

    course = relationship("CourseModel", back_populates="modules")
    lessons = relationship("LessonModel", back_populates="module")


class LessonModel(Base):
    __tablename__ = "lessons"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(Text)
    module_id = Column(Integer, ForeignKey("modules.id"))
    order = Column(Integer)
    video_url = Column(String, nullable=True)
    video_filename = Column(String, nullable=True)
    has_quiz = Column(Boolean, default=True)

    module = relationship("ModuleModel", back_populates="lessons")
    progress = relationship("ProgressModel", back_populates="lesson")
    quiz_attempts = relationship("QuizAttemptModel", back_populates="lesson")


class EnrollmentModel(Base):
    __tablename__ = "enrollments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    course_id = Column(Integer, ForeignKey("courses.id"))
    enrolled_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserModel", back_populates="enrollments")
    course = relationship("CourseModel", back_populates="enrollments")
    progress = relationship("ProgressModel", back_populates="enrollment")


class ProgressModel(Base):
    __tablename__ = "progress"

    id = Column(Integer, primary_key=True, index=True)
    enrollment_id = Column(Integer, ForeignKey("enrollments.id"))
    lesson_id = Column(Integer, ForeignKey("lessons.id"))
    completed = Column(Boolean, default=False)
    completed_at = Column(DateTime, nullable=True)

    enrollment = relationship("EnrollmentModel", back_populates="progress")
    lesson = relationship("LessonModel", back_populates="progress")


class QuizAttemptModel(Base):
    __tablename__ = "quiz_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    lesson_id = Column(Integer, ForeignKey("lessons.id"))
    questions = Column(JSON)
    user_answers = Column(JSON)
    score = Column(Integer)
    passed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserModel", back_populates="quiz_attempts")
    lesson = relationship("LessonModel", back_populates="quiz_attempts")


class CertificateModel(Base):
    __tablename__ = "certificates"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    course_id = Column(Integer, ForeignKey("courses.id"))
    issued_at = Column(DateTime, default=datetime.utcnow)
    certificate_hash = Column(String, unique=True, index=True)
    pdf_filename = Column(String, nullable=True)

    user = relationship("UserModel", back_populates="certificates")
    course = relationship("CourseModel", back_populates="certificates")


class PaymentModel(Base):
    """Tracks PesaPal payment attempts and their status."""
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    course_id = Column(Integer, ForeignKey("courses.id"))
    # Our unique merchant reference sent to PesaPal
    merchant_reference = Column(String, unique=True, index=True)
    # PesaPal's tracking ID (returned after SubmitOrderRequest)
    order_tracking_id = Column(String, nullable=True, index=True)
    amount = Column(Integer, default=COURSE_PRICE_UGX)
    currency = Column(String, default="UGX")
    # PENDING | COMPLETED | FAILED | INVALID
    status = Column(String, default="PENDING")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("UserModel", back_populates="payments")
    course = relationship("CourseModel", back_populates="payments")


# Create tables
Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class UserBase(BaseModel):
    email: EmailStr
    full_name: str


class UserCreate(UserBase):
    password: str
    is_instructor: bool = False


class User(UserBase):
    id: int
    is_active: bool
    is_instructor: bool
    created_at: datetime
    bio: Optional[str] = None
    profile_image_filename: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    phone: Optional[str] = None

    class Config:
        orm_mode = True


# --- NEW: Profile update schema ---
class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    phone: Optional[str] = None


class ProfileResponse(BaseModel):
    id: int
    email: str
    full_name: str
    is_active: bool
    is_instructor: bool
    created_at: datetime
    bio: Optional[str] = None
    profile_image_url: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    phone: Optional[str] = None

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class CourseBase(BaseModel):
    title: str
    description: Optional[str] = None


class CourseCreate(CourseBase):
    pass


class Course(CourseBase):
    id: int
    instructor_id: int
    created_at: datetime
    instructor_name: Optional[str] = None
    is_published: bool
    image_url: Optional[str] = None
    image_filename: Optional[str] = None

    class Config:
        orm_mode = True


class ModuleBase(BaseModel):
    title: str
    description: Optional[str] = None
    order: int


class ModuleCreate(ModuleBase):
    course_id: int


class ModuleCreateRequest(ModuleBase):
    pass


class Module(ModuleBase):
    id: int
    course_id: int

    class Config:
        orm_mode = True


class LessonBase(BaseModel):
    title: str
    content: Optional[str] = None
    order: int
    video_url: Optional[str] = None
    has_quiz: Optional[bool] = True


class LessonCreate(LessonBase):
    pass


class Lesson(LessonBase):
    id: int
    module_id: int
    video_filename: Optional[str] = None

    class Config:
        orm_mode = True


class LessonResponse(BaseModel):
    id: int
    title: str
    content: Optional[str] = None
    order: int
    module_id: int
    video_url: Optional[str] = None
    video_filename: Optional[str] = None
    has_quiz: bool
    completed: Optional[bool] = False

    class Config:
        orm_mode = True


class VideoTokenResponse(BaseModel):
    token: str
    expires_at: datetime


# Quiz Models
class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: int


class QuizRequest(BaseModel):
    lesson_content: str


class QuizResponse(BaseModel):
    questions: List[QuizQuestion]
    quiz_id: str


class QuizSubmission(BaseModel):
    answers: List[int]
    quiz_id: str


class QuizResult(BaseModel):
    score: int
    total: int
    passed: bool
    correct_answers: List[int]
    attempt_id: int
    questions: List[QuizQuestion]


class QuizAttemptResponse(BaseModel):
    id: int
    user_id: int
    lesson_id: int
    score: int
    passed: bool
    created_at: datetime

    class Config:
        orm_mode = True


class QuizQuestionResponse(BaseModel):
    question: str
    options: List[str]
    user_answer: int
    correct_answer: int
    is_correct: bool


class QuizAttemptDetailResponse(BaseModel):
    id: int
    user_id: int
    lesson_id: int
    score: int
    total: int
    passed: bool
    created_at: datetime
    questions: List[QuizQuestionResponse]

    class Config:
        orm_mode = True

class AIConsultRequest(BaseModel):
    question: str


class AIConsultResponse(BaseModel):
    answer: str
    
# Certificate Models
class CertificateResponse(BaseModel):
    id: int
    user_id: int
    course_id: int
    course_title: Optional[str] = None
    issued_at: datetime
    certificate_hash: str
    download_url: Optional[str] = None

    class Config:
        orm_mode = True


class CertificateEligibilityResponse(BaseModel):
    eligible: bool
    completed_lessons: int
    total_lessons: int
    progress_percentage: float
    message: Optional[str] = None
    existing_certificate: Optional[CertificateResponse] = None


# New: image URL response model
class ImageUrlResponse(BaseModel):
    url: str
    expires_in: int  # seconds


# Payment Models
class PaymentInitiateResponse(BaseModel):
    redirect_url: str
    merchant_reference: str
    order_tracking_id: Optional[str] = None


class PaymentStatusResponse(BaseModel):
    merchant_reference: str
    order_tracking_id: Optional[str] = None
    status: str
    amount: int
    currency: str
    course_id: int
    created_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def migrate_has_quiz_default(db: Session):
    """Ensure all lessons have has_quiz = True by default."""
    try:
        logger.info("Starting migration: Setting default has_quiz to True for all lessons")
        total_lessons = db.query(LessonModel).count()
        lessons_without_quiz = db.query(LessonModel).filter(
            (LessonModel.has_quiz == False) | (LessonModel.has_quiz.is_(None))
        ).count()
        logger.info(f"Total lessons: {total_lessons}, Lessons without quiz: {lessons_without_quiz}")
        db.query(LessonModel).filter(
            (LessonModel.has_quiz == False) | (LessonModel.has_quiz.is_(None))
        ).update({LessonModel.has_quiz: True}, synchronize_session=False)
        db.commit()
        lessons_still_without_quiz = db.query(LessonModel).filter(
            (LessonModel.has_quiz == False) | (LessonModel.has_quiz.is_(None))
        ).count()
        logger.info(f"Migration completed. Lessons still without quiz: {lessons_still_without_quiz}")
        return {
            "message": "Migration completed successfully",
            "lessons_updated": lessons_without_quiz,
            "lessons_still_without_quiz": lessons_still_without_quiz,
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Migration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")


# Auth setup
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440       # 24 hours
VIDEO_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="eLearning Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://online-yze5-myir9uasl-dariusmumberes-projects.vercel.app",
        "https://online-coderise.vercel.app",
        "https://elearning-1-fyf2.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user_by_email(db: Session, email: str):
    return db.query(UserModel).filter(UserModel.email == email).first()


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_video_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=VIDEO_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = get_user_by_email(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user


def check_previous_lessons_completed(db: Session, enrollment_id: int, current_lesson: LessonModel):
    """Check if all previous lessons in the module are completed."""
    all_lessons = (
        db.query(LessonModel)
        .filter(LessonModel.module_id == current_lesson.module_id)
        .order_by(LessonModel.order)
        .all()
    )
    current_index = next((i for i, l in enumerate(all_lessons) if l.id == current_lesson.id), None)
    if current_index == 0:
        return True
    for i in range(current_index):
        prev = all_lessons[i]
        progress = db.query(ProgressModel).filter(
            ProgressModel.enrollment_id == enrollment_id,
            ProgressModel.lesson_id == prev.id,
            ProgressModel.completed == True,
        ).first()
        if not progress:
            return False
    return True


# ---------------------------------------------------------------------------
# B2 Helper Functions
# ---------------------------------------------------------------------------

async def upload_to_b2(file: UploadFile, folder: str) -> str:
    """Upload a file to Backblaze B2 and return the object key."""
    try:
        file_extension = file.filename.split(".")[-1]
        filename = f"{folder}/{uuid.uuid4()}.{file_extension}"
        file_content = await file.read()
        b2_client.put_object(
            Bucket=B2_BUCKET_NAME,
            Key=filename,
            Body=file_content,
            ContentType=file.content_type,
        )
        logger.info(f"File uploaded to B2: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error uploading file to B2: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


async def delete_from_b2(filename: str):
    """Delete a file from Backblaze B2."""
    try:
        b2_client.delete_object(Bucket=B2_BUCKET_NAME, Key=filename)
        logger.info(f"File deleted from B2: {filename}")
    except Exception as e:
        logger.error(f"Error deleting file from B2: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


async def generate_presigned_url(filename: str, expiration: int = 3600) -> Optional[str]:
    """Generate a presigned URL for a private B2 object."""
    try:
        if not filename:
            return None
        url = b2_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': B2_BUCKET_NAME, 'Key': filename},
            ExpiresIn=expiration,
        )
        logger.info(f"Generated presigned URL for: {filename}")
        return url
    except ClientError as e:
        logger.error(f"Error generating presigned URL: {e}")
        return None


def _resolve_image_filename(course: CourseModel) -> Optional[str]:
    if course.image_filename:
        return course.image_filename
    if course.image_url:
        if not course.image_url.startswith("http") and not course.image_url.startswith("/"):
            return course.image_url
        if course.image_url.startswith("/b2-proxy/"):
            return course.image_url[len("/b2-proxy/"):]
        if course.image_url.startswith("http"):
            from urllib.parse import urlparse
            path = urlparse(course.image_url).path
            path = path.lstrip("/")
            if path:
                return path
    return None


async def get_course_image_url(course: CourseModel) -> Optional[str]:
    key = _resolve_image_filename(course)
    if not key:
        return None
    return f"/b2-proxy/{key}"


# ---------------------------------------------------------------------------
# Quiz Cache Helpers
# ---------------------------------------------------------------------------

def get_quiz_cache_key(user_id: int, lesson_id: int, quiz_id: str = None):
    if quiz_id:
        return f"quiz:{user_id}:{lesson_id}:{quiz_id}"
    return f"quiz:{user_id}:{lesson_id}"


def cache_quiz_questions(user_id: int, lesson_id: int, questions: List[dict], quiz_id: str):
    cache_key = get_quiz_cache_key(user_id, lesson_id, quiz_id)
    try:
        if redis_client:
            redis_client.setex(cache_key, 3600, pickle.dumps(questions))
        else:
            quiz_cache[cache_key] = {
                'questions': questions,
                'expires': datetime.utcnow() + timedelta(hours=1),
            }
        logger.info(f"Cached quiz questions for user {user_id}, lesson {lesson_id}, quiz_id {quiz_id}")
    except Exception as e:
        logger.error(f"Error caching quiz questions: {e}")


def get_cached_quiz_questions(user_id: int, lesson_id: int, quiz_id: str):
    cache_key = get_quiz_cache_key(user_id, lesson_id, quiz_id)
    try:
        if redis_client:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)
        else:
            if cache_key in quiz_cache:
                cached_item = quiz_cache[cache_key]
                if cached_item['expires'] > datetime.utcnow():
                    return cached_item['questions']
                else:
                    del quiz_cache[cache_key]
    except Exception as e:
        logger.error(f"Error retrieving cached quiz questions: {e}")
    return None


# ---------------------------------------------------------------------------
# PesaPal Helpers
# ---------------------------------------------------------------------------

def pesapal_get_token() -> str:
    """
    Authenticate with PesaPal and return a bearer token.
    Tokens are cached in memory and reused until they expire.
    """
    global _pesapal_token_cache

    now = datetime.utcnow()
    cached_token = _pesapal_token_cache.get("token")
    cached_expiry = _pesapal_token_cache.get("expiry")

    if cached_token and cached_expiry and now < cached_expiry:
        logger.info(f"Using cached PesaPal token (expires at {cached_expiry})")
        return cached_token

    auth_url = f"{PESAPAL_BASE_URL}/api/Auth/RequestToken"
    payload = {
        "consumer_key": PESAPAL_CONSUMER_KEY,
        "consumer_secret": PESAPAL_CONSUMER_SECRET,
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    logger.info(f"Requesting PesaPal token from {auth_url}")
    logger.debug(f"Auth payload: {payload}")

    try:
        resp = requests.post(auth_url, json=payload, headers=headers, timeout=30)
        logger.info(f"PesaPal auth response status: {resp.status_code}")
        logger.info(f"PesaPal auth response headers: {dict(resp.headers)}")
        
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"PesaPal auth response body: {json.dumps(data, indent=2)}")
    except requests.RequestException as e:
        logger.error(f"PesaPal auth error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response body: {e.response.text}")
        raise HTTPException(status_code=502, detail="Could not authenticate with PesaPal.")

    token = data.get("token")
    expiry_str = data.get("expiryDate")

    if not token:
        logger.error(f"PesaPal auth response missing token: {data}")
        raise HTTPException(status_code=502, detail="Invalid PesaPal authentication response.")

    # Parse expiry; default to 50 minutes from now if not parseable
    try:
        expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00")).replace(tzinfo=None)
        # Buffer of 5 minutes
        expiry -= timedelta(minutes=5)
        logger.info(f"PesaPal token expires at {expiry}")
    except Exception as e:
        logger.warning(f"Could not parse expiry date '{expiry_str}': {e}")
        expiry = now + timedelta(minutes=50)

    _pesapal_token_cache = {"token": token, "expiry": expiry}
    logger.info("PesaPal token obtained and cached.")
    return token


def pesapal_register_ipn(token: str) -> str:
    """
    Register the IPN URL with PesaPal and return the ipn_id.
    Only registers once per process lifetime; subsequent calls return cached ID.
    """
    global _pesapal_ipn_id

    # Return cached IPN ID if available
    if _pesapal_ipn_id:
        logger.info(f"Using cached IPN ID: {_pesapal_ipn_id}")
        return _pesapal_ipn_id

    # Check env-configured IPN ID (set after first registration)
    env_ipn_id = os.getenv("PESAPAL_IPN_ID")
    if env_ipn_id:
        logger.info(f"Using env-configured IPN ID: {env_ipn_id}")
        _pesapal_ipn_id = env_ipn_id
        return _pesapal_ipn_id

    ipn_url = f"{PESAPAL_BASE_URL}/api/URLSetup/RegisterIPN"
    payload = {
        "url": PESAPAL_IPN_URL,
        "ipn_notification_type": "GET",
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    logger.info(f"Registering IPN URL: {PESAPAL_IPN_URL}")
    logger.info(f"IPN registration request to {ipn_url}")
    logger.debug(f"IPN payload: {payload}")

    try:
        resp = requests.post(ipn_url, json=payload, headers=headers, timeout=30)
        logger.info(f"IPN registration response status: {resp.status_code}")
        logger.info(f"IPN registration response headers: {dict(resp.headers)}")
        
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"IPN registration response body: {json.dumps(data, indent=2)}")
    except requests.RequestException as e:
        logger.error(f"PesaPal IPN registration error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response body: {e.response.text}")
        raise HTTPException(status_code=502, detail="Could not register IPN with PesaPal.")

    ipn_id = data.get("ipn_id")
    if not ipn_id:
        logger.error(f"PesaPal IPN registration response missing ipn_id: {data}")
        raise HTTPException(status_code=502, detail="Invalid PesaPal IPN registration response.")

    _pesapal_ipn_id = ipn_id
    logger.info(f"PesaPal IPN registered. ipn_id={ipn_id}")
    return ipn_id


def pesapal_submit_order(
    token: str,
    ipn_id: str,
    merchant_reference: str,
    amount: int,
    currency: str,
    description: str,
    email: str,
    first_name: str,
    last_name: str,
    callback_url: str,
) -> dict:
    """
    Submit an order to PesaPal and return the response dict containing
    order_tracking_id and redirect_url.
    """
    order_url = f"{PESAPAL_BASE_URL}/api/Transactions/SubmitOrderRequest"
    payload = {
        "id": merchant_reference,
        "currency": currency,
        "amount": float(amount),
        "description": description[:100],
        "callback_url": callback_url,
        "notification_id": ipn_id,
        "billing_address": {
            "email_address": email,
            "first_name": first_name,
            "last_name": last_name,
            "phone_number": "",
            "country_code": "UG",
            "line_1": "",
            "line_2": "",
            "city": "",
            "state": "",
            "postal_code": "",
            "zip_code": "",
        },
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    logger.info(f"Submitting order to PesaPal: {order_url}")
    logger.info(f"Order payload: {json.dumps(payload, indent=2)}")
    logger.info(f"Callback URL: {callback_url}")

    try:
        resp = requests.post(order_url, json=payload, headers=headers, timeout=30)
        logger.info(f"SubmitOrder response status: {resp.status_code}")
        logger.info(f"SubmitOrder response headers: {dict(resp.headers)}")
        
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"SubmitOrder response body (FULL): {json.dumps(data, indent=2)}")
    except requests.RequestException as e:
        logger.error(f"PesaPal SubmitOrderRequest error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response body: {e.response.text}")
        raise HTTPException(status_code=502, detail="Could not submit payment order to PesaPal.")

    # -----------------------------------------------------------------------
    # Robust error handling — data['error'] can be None, a dict, or absent.
    # -----------------------------------------------------------------------
    error_field = data.get("error")
    if error_field is not None:
        # error_field could be a dict with a 'message' key, or some other truthy value
        if isinstance(error_field, dict):
            error_message = error_field.get("message") or error_field.get("errorMessage") or str(error_field)
        else:
            error_message = str(error_field)
        logger.error(f"PesaPal order error field: {error_field}")
        raise HTTPException(
            status_code=502,
            detail=f"PesaPal error: {error_message}",
        )

    # Also check for a top-level status indicating failure
    # PesaPal sometimes returns {"status": "200", ...} on success
    status_val = data.get("status")
    if status_val and str(status_val) not in ("200", "0"):
        # status "0" sometimes means success in PesaPal v3
        logger.warning(f"PesaPal unexpected status value: {status_val}")

    redirect_url = data.get("redirect_url")
    if not redirect_url:
        logger.error(f"PesaPal response missing redirect_url: {data}")
        raise HTTPException(
            status_code=502,
            detail="PesaPal did not return a payment URL. Please try again.",
        )

    logger.info(f"PesaPal order submitted successfully. Redirect URL: {redirect_url}")
    return data


def pesapal_get_transaction_status(token: str, order_tracking_id: str) -> dict:
    """Query PesaPal for the current status of a transaction."""
    status_url = f"{PESAPAL_BASE_URL}/api/Transactions/GetTransactionStatus"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    params = {"orderTrackingId": order_tracking_id}

    logger.info(f"Querying transaction status for {order_tracking_id}")
    logger.info(f"Status URL: {status_url}")
    logger.debug(f"Query params: {params}")

    try:
        resp = requests.get(status_url, headers=headers, params=params, timeout=30)
        logger.info(f"Transaction status response status: {resp.status_code}")
        logger.info(f"Transaction status response headers: {dict(resp.headers)}")
        
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Transaction status response body (FULL): {json.dumps(data, indent=2)}")
        return data
    except requests.RequestException as e:
        logger.error(f"PesaPal GetTransactionStatus error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response body: {e.response.text}")
        raise HTTPException(status_code=502, detail="Could not query payment status from PesaPal.")


def _enroll_user_in_course(db: Session, user_id: int, course_id: int):
    """Create an enrollment record if one does not already exist."""
    existing = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == user_id,
        EnrollmentModel.course_id == course_id,
    ).first()
    if not existing:
        enrollment = EnrollmentModel(user_id=user_id, course_id=course_id)
        db.add(enrollment)
        db.commit()
        logger.info(f"User {user_id} enrolled in course {course_id} after payment.")
    else:
        logger.info(f"User {user_id} already enrolled in course {course_id}")


# ---------------------------------------------------------------------------
# Quiz Generation
# ---------------------------------------------------------------------------

async def generate_quiz(lesson_content: str) -> List[QuizQuestion]:
    """Generate a quiz using Gemini AI based on lesson content."""
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY not set, using fallback quiz")
            return create_fallback_quiz(lesson_content)

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
        Based on the following lesson content, generate exactly 5 multiple-choice questions with 4 options each.
        The questions should test understanding of key concepts from the lesson.
        Return the questions in JSON format with this structure:
        {{
            "questions": [
                {{
                    "question": "Question text",
                    "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                    "correct_answer": 0
                }}
            ]
        }}
        The correct_answer should be the index (0-3) of the correct option.

        IMPORTANT: Make sure the questions are directly related to the content and test actual understanding.

        Lesson Content:
        {lesson_content}

        Return ONLY valid JSON. Do not include any additional text.
        """

        response = model.generate_content(prompt)

        try:
            json_str = response.text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            json_str = json_str.strip()

            quiz_data = json.loads(json_str)

            if "questions" in quiz_data and len(quiz_data["questions"]) == 5:
                validated_questions = []
                for i, question in enumerate(quiz_data["questions"]):
                    if not all(key in question for key in ["question", "options", "correct_answer"]):
                        raise ValueError(f"Question {i+1} missing required fields")
                    if len(question["options"]) != 4:
                        raise ValueError(f"Question {i+1} doesn't have exactly 4 options")
                    if not isinstance(question["correct_answer"], int):
                        raise ValueError(f"Question {i+1} correct_answer is not integer")
                    if not 0 <= question["correct_answer"] <= 3:
                        raise ValueError(f"Question {i+1} has invalid correct_answer index")
                    question["options"] = [str(opt) for opt in question["options"]]
                    validated_questions.append(question)
                return validated_questions
            else:
                raise ValueError("Invalid quiz format from AI")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse AI response: {e}")
            logger.error(f"AI response: {response.text}")
            return create_fallback_quiz(lesson_content)

    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        return create_fallback_quiz("")


def create_fallback_quiz(lesson_content: str) -> List[dict]:
    key_concepts = []
    if lesson_content:
        sentences = lesson_content.split('.')[:3]
        key_concepts = [s.strip() for s in sentences if s.strip()]

    questions = []
    for i in range(5):
        if key_concepts and i < len(key_concepts):
            question_text = f"What is the main idea about: {key_concepts[i]}"
        else:
            question_text = f"Question {i+1} about the lesson content"
        questions.append({
            "question": question_text,
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": 0,
        })
    return questions


# ---------------------------------------------------------------------------
# Certificate Helpers
# ---------------------------------------------------------------------------

def generate_certificate_hash(user_id: int, course_id: int) -> str:
    unique_string = f"{user_id}_{course_id}_{datetime.utcnow().isoformat()}_{uuid.uuid4()}"
    return hashlib.sha256(unique_string.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Modern Certificate PDF
# ---------------------------------------------------------------------------

class _HRule(Flowable):
    """A thin horizontal rule with optional color."""
    def __init__(self, width, thickness=1, color=colors.black):
        super().__init__()
        self.width = width
        self.thickness = thickness
        self.color = color
        self.height = thickness + 2

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.thickness / 2, self.width, self.thickness / 2)


def _draw_certificate_background(canvas, doc):
    """Draw the decorative page background, borders, and watermark elements."""
    canvas.saveState()
    W, H = A4  # 595.27 x 841.89 pts

    # ── Deep navy background ──────────────────────────────────────────────
    canvas.setFillColor(colors.HexColor('#0B1A2E'))
    canvas.rect(0, 0, W, H, fill=1, stroke=0)

    # ── Top gold band ─────────────────────────────────────────────────────
    canvas.setFillColor(colors.HexColor('#C9A84C'))
    canvas.rect(0, H - 58, W, 58, fill=1, stroke=0)

    # ── Bottom gold band ──────────────────────────────────────────────────
    canvas.setFillColor(colors.HexColor('#C9A84C'))
    canvas.rect(0, 0, W, 58, fill=1, stroke=0)

    # ── Thin white accent lines just inside the bands ─────────────────────
    canvas.setStrokeColor(colors.white)
    canvas.setLineWidth(1)
    canvas.line(0, H - 62, W, H - 62)
    canvas.line(0, 62, W, 62)

    # ── Side accent bars (narrow gold) ────────────────────────────────────
    canvas.setFillColor(colors.HexColor('#C9A84C'))
    canvas.rect(0, 58, 14, H - 116, fill=1, stroke=0)
    canvas.rect(W - 14, 58, 14, H - 116, fill=1, stroke=0)

    # ── Inner white accent lines beside the side bars ─────────────────────
    canvas.setStrokeColor(colors.HexColor('#FFFFFF'))
    canvas.setLineWidth(0.5)
    canvas.line(18, 62, 18, H - 62)
    canvas.line(W - 18, 62, W - 18, H - 62)

    # ── Decorative corner diamonds (top-left, top-right, bottom-left, bottom-right)
    gold = colors.HexColor('#C9A84C')
    white = colors.white
    corners = [(14, H - 58), (W - 14, H - 58), (14, 58), (W - 14, 58)]
    for cx, cy in corners:
        # Outer diamond
        canvas.setFillColor(colors.HexColor('#0B1A2E'))
        canvas.setStrokeColor(gold)
        canvas.setLineWidth(1.5)
        size = 10
        p = canvas.beginPath()
        p.moveTo(cx, cy + size)
        p.lineTo(cx + size, cy)
        p.lineTo(cx, cy - size)
        p.lineTo(cx - size, cy)
        p.close()
        canvas.drawPath(p, fill=1, stroke=1)
        # Inner dot
        canvas.setFillColor(gold)
        canvas.circle(cx, cy, 3, fill=1, stroke=0)

    # ── Watermark seal (large faint circle in centre) ─────────────────────
    canvas.setStrokeColor(colors.HexColor('#1A2E4A'))
    canvas.setLineWidth(1)
    cx, cy = W / 2, H / 2
    for r in (100, 108, 116):
        canvas.circle(cx, cy, r, fill=0, stroke=1)

    # Star points around outer ring
    canvas.setStrokeColor(colors.HexColor('#1A2E4A'))
    canvas.setLineWidth(0.5)
    for i in range(24):
        angle = math.radians(i * 15)
        x1 = cx + 112 * math.cos(angle)
        y1 = cy + 112 * math.sin(angle)
        x2 = cx + 120 * math.cos(angle)
        y2 = cy + 120 * math.sin(angle)
        canvas.line(x1, y1, x2, y2)

    # ── Platform name in gold inside the bands ────────────────────────────
    canvas.setFillColor(colors.HexColor('#0B1A2E'))
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawCentredString(W / 2, H - 38, "C O D E R I S E   A C A D E M Y")

    canvas.setFillColor(colors.HexColor('#0B1A2E'))
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(W / 2, 22, "online-coderise.vercel.app  ·  Empowering Learners Worldwide")

    canvas.restoreState()


def create_certificate_pdf(user: UserModel, course: CourseModel, certificate_hash: str) -> BytesIO:
    """Create a modern, premium certificate PDF."""
    buffer = BytesIO()

    # Page margins: inside the decorative side bars (14 pt) + padding
    left_margin = 38
    right_margin = 38
    top_margin = 72      # clear the top gold band (58) + a bit
    bottom_margin = 72   # clear the bottom gold band

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
    )

    W, H = A4
    usable_width = W - left_margin - right_margin   # ~519 pt

    # ── Colour palette ────────────────────────────────────────────────────
    GOLD      = colors.HexColor('#C9A84C')
    LIGHT_GOLD = colors.HexColor('#E8D5A3')
    WHITE     = colors.white
    LIGHT_BLUE = colors.HexColor('#A8C8E8')
    SILVER    = colors.HexColor('#C0C0C0')

    # ── Paragraph styles ──────────────────────────────────────────────────
    def _style(name, font='Helvetica', size=12, color=WHITE, space_before=0, space_after=8, align=TA_CENTER, leading=None):
        return ParagraphStyle(
            name,
            fontName=font,
            fontSize=size,
            textColor=color,
            spaceBefore=space_before,
            spaceAfter=space_after,
            alignment=align,
            leading=leading or size * 1.2,
        )

    style_eyebrow  = _style('eyebrow',  'Helvetica',      9,  LIGHT_GOLD,   0,  4,  TA_CENTER)
    style_big_title= _style('bigtitle', 'Helvetica-Bold', 30, GOLD,         0, 10,  TA_CENTER, 34)
    style_subtitle = _style('subtitle', 'Helvetica',      11, LIGHT_BLUE,   0,  6,  TA_CENTER)
    style_present  = _style('present',  'Helvetica',      11, SILVER,       6,  4,  TA_CENTER)
    style_name     = _style('name',     'Helvetica-Bold', 28, WHITE,        4, 10,  TA_CENTER, 32)
    style_for_comp = _style('forcomp',  'Helvetica',      10, SILVER,       0,  4,  TA_CENTER)
    style_course   = _style('course',   'Helvetica-Bold', 18, GOLD,         4, 10,  TA_CENTER, 22)
    style_body     = _style('body',     'Helvetica',      10, SILVER,       4,  4,  TA_CENTER)
    style_date     = _style('date',     'Helvetica',      10, LIGHT_GOLD,   2,  2,  TA_CENTER)
    style_hash     = _style('hash',     'Helvetica',       7, colors.HexColor('#5A7A9A'), 8, 0, TA_CENTER)
    style_sig_name = _style('signame',  'Helvetica-Bold', 10, WHITE,        2,  0,  TA_CENTER)
    style_sig_role = _style('sigrole',  'Helvetica',       8, SILVER,       0,  0,  TA_CENTER)

    elements = []

    # ── Logo (attempt remote fetch; fall back to text) ────────────────────
    try:
        logo_resp = requests.get(
            "https://raw.githubusercontent.com/Dariusmumbere/elearning/main/logo.png",
            timeout=5
        )
        if logo_resp.status_code == 200:
            logo_img = Image(BytesIO(logo_resp.content))
            aspect = logo_img.imageWidth / logo_img.imageHeight
            logo_img.drawWidth  = 0.7 * inch
            logo_img.drawHeight = 0.7 * inch / aspect
            logo_img.hAlign = 'CENTER'
            elements.append(logo_img)
            elements.append(Spacer(1, 6))
    except Exception as e:
        logger.warning(f"Could not load logo for certificate: {e}")

    # ── Eye-brow text ─────────────────────────────────────────────────────
    elements.append(Paragraph("— OFFICIAL CERTIFICATE —", style_eyebrow))
    elements.append(Spacer(1, 4))

    # ── Main title ────────────────────────────────────────────────────────
    elements.append(Paragraph("Certificate of Achievement", style_big_title))

    # ── Gold divider ──────────────────────────────────────────────────────
    elements.append(Spacer(1, 6))
    elements.append(_HRule(usable_width, thickness=2, color=GOLD))
    elements.append(Spacer(1, 2))
    elements.append(_HRule(usable_width, thickness=0.5, color=LIGHT_GOLD))
    elements.append(Spacer(1, 12))

    # ── Subtitle ──────────────────────────────────────────────────────────
    elements.append(Paragraph("This is to proudly certify that", style_present))
    elements.append(Spacer(1, 8))

    # ── Recipient name in a styled box ───────────────────────────────────
    name_table = Table(
        [[Paragraph(user.full_name, style_name)]],
        colWidths=[usable_width],
    )
    name_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#102040')),
        ('BOX',        (0, 0), (-1, -1), 1.5, GOLD),
        ('LEFTPADDING',  (0, 0), (-1, -1), 18),
        ('RIGHTPADDING', (0, 0), (-1, -1), 18),
        ('TOPPADDING',   (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 10),
        ('ALIGN',        (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(name_table)
    elements.append(Spacer(1, 14))

    # ── Completion line ───────────────────────────────────────────────────
    elements.append(Paragraph("has successfully completed all requirements for the course", style_for_comp))
    elements.append(Spacer(1, 8))

    # ── Course title in gold accent box ───────────────────────────────────
    course_table = Table(
        [[Paragraph(course.title, style_course)]],
        colWidths=[usable_width],
    )
    course_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0D2240')),
        ('BOX',        (0, 0), (-1, -1), 0.75, LIGHT_GOLD),
        ('LINEBELOW',  (0, 0), (-1, -1), 3, GOLD),
        ('LEFTPADDING',  (0, 0), (-1, -1), 24),
        ('RIGHTPADDING', (0, 0), (-1, -1), 24),
        ('TOPPADDING',   (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 12),
        ('ALIGN',        (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(course_table)
    elements.append(Spacer(1, 14))

    # ── Body copy ─────────────────────────────────────────────────────────
    elements.append(Paragraph(
        "demonstrating dedication, commitment, and mastery of the curriculum.",
        style_body
    ))
    elements.append(Spacer(1, 4))

    # ── Issue date ────────────────────────────────────────────────────────
    completion_date = datetime.utcnow().strftime("%B %d, %Y")
    elements.append(Paragraph(f"Issued on  {completion_date}", style_date))
    elements.append(Spacer(1, 16))

    # ── Second divider ────────────────────────────────────────────────────
    elements.append(_HRule(usable_width, thickness=0.5, color=LIGHT_GOLD))
    elements.append(Spacer(1, 2))
    elements.append(_HRule(usable_width, thickness=2, color=GOLD))
    elements.append(Spacer(1, 18))

    # ── Signature row ─────────────────────────────────────────────────────
    col = usable_width / 3

    def _sig_cell(name, role):
        return [
            Paragraph("______________________", _style('sigline', 'Helvetica', 10, SILVER, 0, 2, TA_CENTER)),
            Paragraph(name, style_sig_name),
            Paragraph(role, style_sig_role),
        ]

    sig_data = [[
        _sig_cell("Course Instructor", "Lead Instructor"),
        _sig_cell("CodeRise Academy", "Issuing Authority"),
        _sig_cell("Academic Director", "Platform Director"),
    ]]
    sig_table = Table(sig_data, colWidths=[col, col, col])
    sig_table.setStyle(TableStyle([
        ('ALIGN',   (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',  (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',  (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(sig_table)
    elements.append(Spacer(1, 16))

    # ── Certificate ID footer ─────────────────────────────────────────────
    id_table = Table(
        [[Paragraph(f"Certificate ID: {certificate_hash.upper()}  ·  Verify at online-coderise.vercel.app/verify", style_hash)]],
        colWidths=[usable_width],
    )
    id_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#061220')),
        ('BOX',        (0, 0), (-1, -1), 0.5, colors.HexColor('#1E3A5A')),
        ('TOPPADDING',   (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 6),
        ('LEFTPADDING',  (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(id_table)

    # ── Build PDF ─────────────────────────────────────────────────────────
    doc.build(
        elements,
        onFirstPage=_draw_certificate_background,
        onLaterPages=_draw_certificate_background,
    )
    buffer.seek(0)
    return buffer


async def upload_certificate_to_b2(pdf_buffer: BytesIO, filename: str) -> str:
    try:
        b2_client.put_object(
            Bucket=B2_BUCKET_NAME,
            Key=filename,
            Body=pdf_buffer.getvalue(),
            ContentType='application/pdf',
        )
        logger.info(f"Certificate uploaded to B2: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error uploading certificate to B2: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading certificate: {str(e)}")


def check_course_completion(db: Session, user_id: int, course_id: int) -> tuple:
    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    if not course:
        return False, 0, 0, 0
    total_lessons = 0
    completed_lessons = 0
    for module in course.modules:
        total_lessons += len(module.lessons)
        for lesson in module.lessons:
            progress = db.query(ProgressModel).join(EnrollmentModel).filter(
                EnrollmentModel.user_id == user_id,
                EnrollmentModel.course_id == course_id,
                ProgressModel.lesson_id == lesson.id,
                ProgressModel.completed == True,
            ).first()
            if progress:
                completed_lessons += 1
    progress_percentage = (completed_lessons / total_lessons * 100) if total_lessons > 0 else 0
    return completed_lessons == total_lessons, completed_lessons, total_lessons, progress_percentage


def _build_course_response(course: CourseModel, instructor_name: str, image_url: Optional[str]) -> dict:
    return {
        "id": course.id,
        "title": course.title,
        "description": course.description,
        "instructor_id": course.instructor_id,
        "instructor_name": instructor_name,
        "created_at": course.created_at,
        "is_published": course.is_published,
        "image_url": image_url,
        "image_filename": course.image_filename,
    }


# ===========================================================================
# Routes
# ===========================================================================

# ---------------------------------------------------------------------------
# PesaPal Payment Routes
# ---------------------------------------------------------------------------

@app.post("/payments/initiate/{course_id}", response_model=PaymentInitiateResponse)
async def initiate_payment(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Initiate a PesaPal payment for a course enrollment.
    Returns the PesaPal redirect URL where the user completes payment.
    """
    logger.info(f"=== PAYMENT INITIATION START ===")
    logger.info(f"Course ID: {course_id}, User: {current_user.email} (ID: {current_user.id})")
    
    # Verify course exists and is published
    course = db.query(CourseModel).filter(
        CourseModel.id == course_id,
        CourseModel.is_published == True,
    ).first()
    if not course:
        logger.error(f"Course {course_id} not found or not published")
        raise HTTPException(status_code=404, detail="Course not found.")

    logger.info(f"Course found: {course.title}")

    # Check if user is already enrolled
    existing_enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == course_id,
    ).first()
    if existing_enrollment:
        logger.warning(f"User {current_user.id} already enrolled in course {course_id}")
        raise HTTPException(status_code=400, detail="You are already enrolled in this course.")

    # Check for a pending/completed payment that hasn't been applied yet
    existing_completed_payment = db.query(PaymentModel).filter(
        PaymentModel.user_id == current_user.id,
        PaymentModel.course_id == course_id,
        PaymentModel.status == "COMPLETED",
    ).first()
    if existing_completed_payment:
        logger.info(f"Found completed payment for user {current_user.id}, course {course_id}. Applying enrollment.")
        _enroll_user_in_course(db, current_user.id, course_id)
        raise HTTPException(
            status_code=400,
            detail="Payment already completed. You have been enrolled in the course."
        )

    # Generate a unique merchant reference (max 50 chars, alphanumeric + dash)
    merchant_reference = f"crs-{course_id}-usr-{current_user.id}-{uuid.uuid4().hex[:8]}"
    logger.info(f"Generated merchant reference: {merchant_reference}")

    # Authenticate with PesaPal
    logger.info("Getting PesaPal token...")
    pesapal_token = pesapal_get_token()
    logger.info("PesaPal token obtained successfully")

    # Register IPN (idempotent — cached after first registration)
    logger.info("Registering IPN...")
    ipn_id = pesapal_register_ipn(pesapal_token)
    logger.info(f"IPN registered with ID: {ipn_id}")

    # Build the callback URL with merchant reference so we can identify on return
    callback_url = f"{PESAPAL_CALLBACK_URL}?merchant_reference={merchant_reference}"
    logger.info(f"Callback URL: {callback_url}")

    # Split full name for billing address
    name_parts = current_user.full_name.strip().split(" ", 1)
    first_name = name_parts[0]
    last_name = name_parts[1] if len(name_parts) > 1 else ""
    logger.info(f"Billing info: Name={first_name} {last_name}, Email={current_user.email}")

    # Submit order to PesaPal
    logger.info("Submitting order to PesaPal...")
    order_data = pesapal_submit_order(
        token=pesapal_token,
        ipn_id=ipn_id,
        merchant_reference=merchant_reference,
        amount=COURSE_PRICE_UGX,
        currency="UGX",
        description=f"Enrollment: {course.title}"[:100],
        email=current_user.email,
        first_name=first_name,
        last_name=last_name,
        callback_url=callback_url,
    )

    order_tracking_id = order_data.get("order_tracking_id")
    redirect_url = order_data.get("redirect_url")

    logger.info(f"Order submitted. Tracking ID: {order_tracking_id}")
    logger.info(f"Redirect URL: {redirect_url}")

    if not redirect_url:
        logger.error(f"PesaPal response missing redirect_url: {order_data}")
        raise HTTPException(status_code=502, detail="PesaPal did not return a payment URL.")

    # Persist payment record
    payment = PaymentModel(
        user_id=current_user.id,
        course_id=course_id,
        merchant_reference=merchant_reference,
        order_tracking_id=order_tracking_id,
        amount=COURSE_PRICE_UGX,
        currency="UGX",
        status="PENDING",
    )
    db.add(payment)
    db.commit()
    db.refresh(payment)
    logger.info(f"Payment record created with ID: {payment.id}")

    logger.info(f"=== PAYMENT INITIATION COMPLETE ===")

    return PaymentInitiateResponse(
        redirect_url=redirect_url,
        merchant_reference=merchant_reference,
        order_tracking_id=order_tracking_id,
    )


@app.get("/payments/callback")
async def payment_callback(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Handles the redirect back from PesaPal after the user completes or cancels payment.
    PesaPal appends orderTrackingId and merchant_reference as query params.
    This endpoint verifies the payment status and enrolls the user if successful.
    Redirects the user to the frontend with the result.
    """
    logger.info(f"=== PAYMENT CALLBACK RECEIVED ===")
    
    # Log ALL request details
    logger.info(f"Callback URL: {request.url}")
    logger.info(f"Callback method: {request.method}")
    logger.info(f"Callback headers: {dict(request.headers)}")
    
    params = dict(request.query_params)
    logger.info(f"Callback query parameters (FULL): {json.dumps(params, indent=2)}")
    
    order_tracking_id = params.get("OrderTrackingId") or params.get("orderTrackingId")
    merchant_reference = params.get("OrderMerchantReference") or params.get("merchant_reference")
    
    logger.info(f"Extracted - Tracking ID: {order_tracking_id}")
    logger.info(f"Extracted - Merchant Reference: {merchant_reference}")

    if not merchant_reference:
        logger.error("No merchant_reference found in callback query params")
        redirect_url = f"{PESAPAL_CALLBACK_URL.split('/payment')[0]}/payment/result?status=error&message=Missing+payment+reference"
        logger.info(f"Redirecting to: {redirect_url}")
        return RedirectResponse(url=redirect_url, status_code=302)

    # Lookup payment record
    payment = db.query(PaymentModel).filter(
        PaymentModel.merchant_reference == merchant_reference
    ).first()

    if not payment:
        logger.error(f"No payment found for merchant_reference: {merchant_reference}")
        redirect_url = f"{PESAPAL_CALLBACK_URL.split('/payment')[0]}/payment/result?status=error&message=Payment+not+found"
        return RedirectResponse(url=redirect_url, status_code=302)

    logger.info(f"Found payment record: ID={payment.id}, status={payment.status}")

    # Update order_tracking_id if we didn't have it yet
    if order_tracking_id and not payment.order_tracking_id:
        logger.info(f"Updating payment record with tracking ID: {order_tracking_id}")
        payment.order_tracking_id = order_tracking_id
        db.commit()

    tracking_id = payment.order_tracking_id or order_tracking_id
    logger.info(f"Using tracking ID: {tracking_id}")

    if not tracking_id:
        logger.warning(f"No tracking ID available for payment {payment.id}")
        redirect_url = f"{PESAPAL_CALLBACK_URL.split('/payment')[0]}/payment/result?status=pending&course_id={payment.course_id}"
        return RedirectResponse(url=redirect_url, status_code=302)

    # Query PesaPal for actual transaction status
    try:
        logger.info(f"Querying transaction status from PesaPal for tracking ID: {tracking_id}")
        pesapal_token = pesapal_get_token()
        status_data = pesapal_get_transaction_status(pesapal_token, tracking_id)
        
        payment_status_code = status_data.get("payment_status_description", "").upper()
        pesapal_status = status_data.get("status_code")  # 1=COMPLETED, 0=INVALID, 2=FAILED, etc.
        
        logger.info(f"PesaPal payment_status_description: {payment_status_code}")
        logger.info(f"PesaPal status_code: {pesapal_status}")
        logger.info(f"Full status data: {json.dumps(status_data, indent=2)}")

        if payment_status_code == "COMPLETED" or pesapal_status == 1:
            logger.info(f"Payment {merchant_reference} is COMPLETED!")
            payment.status = "COMPLETED"
            db.commit()
            # Enroll the user
            _enroll_user_in_course(db, payment.user_id, payment.course_id)
            redirect_url = f"{PESAPAL_CALLBACK_URL.split('/payment')[0]}/payment/result?status=success&course_id={payment.course_id}"
            logger.info(f"Redirecting to: {redirect_url}")
            return RedirectResponse(url=redirect_url, status_code=302)
            
        elif payment_status_code in ("FAILED", "INVALID") or pesapal_status in (0, 2):
            logger.warning(f"Payment {merchant_reference} FAILED/INVALID")
            payment.status = "FAILED"
            db.commit()
            redirect_url = f"{PESAPAL_CALLBACK_URL.split('/payment')[0]}/payment/result?status=failed&course_id={payment.course_id}"
            logger.info(f"Redirecting to: {redirect_url}")
            return RedirectResponse(url=redirect_url, status_code=302)
        else:
            logger.info(f"Payment {merchant_reference} status: {payment_status_code} - PENDING")
            redirect_url = f"{PESAPAL_CALLBACK_URL.split('/payment')[0]}/payment/result?status=pending&course_id={payment.course_id}"
            return RedirectResponse(url=redirect_url, status_code=302)

    except Exception as e:
        logger.error(f"Error verifying payment on callback: {e}", exc_info=True)
        redirect_url = f"{PESAPAL_CALLBACK_URL.split('/payment')[0]}/payment/result?status=pending&course_id={payment.course_id}"
        return RedirectResponse(url=redirect_url, status_code=302)


@app.get("/payments/ipn")
async def payment_ipn(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Instant Payment Notification (IPN) endpoint.
    PesaPal calls this endpoint (GET) whenever a payment status changes.
    We query PesaPal for the latest status and update our records.
    """
    logger.info(f"=== IPN RECEIVED ===")
    
    # Log ALL request details
    logger.info(f"IPN URL: {request.url}")
    logger.info(f"IPN method: {request.method}")
    logger.info(f"IPN headers: {dict(request.headers)}")
    
    params = dict(request.query_params)
    logger.info(f"IPN query parameters (FULL): {json.dumps(params, indent=2)}")
    
    order_tracking_id = params.get("orderTrackingId") or params.get("OrderTrackingId")
    merchant_reference = params.get("orderMerchantReference") or params.get("OrderMerchantReference")
    order_notification_type = params.get("orderNotificationType")
    
    logger.info(f"IPN - orderTrackingId: {order_tracking_id}")
    logger.info(f"IPN - merchant_reference: {merchant_reference}")
    logger.info(f"IPN - orderNotificationType: {order_notification_type}")

    if not order_tracking_id:
        logger.warning("IPN received without orderTrackingId")
        return {"status": "ignored", "reason": "missing orderTrackingId"}

    # Lookup by tracking ID or merchant reference
    payment = None
    if merchant_reference:
        logger.info(f"Looking up payment by merchant_reference: {merchant_reference}")
        payment = db.query(PaymentModel).filter(
            PaymentModel.merchant_reference == merchant_reference
        ).first()
        
    if not payment and order_tracking_id:
        logger.info(f"Looking up payment by order_tracking_id: {order_tracking_id}")
        payment = db.query(PaymentModel).filter(
            PaymentModel.order_tracking_id == order_tracking_id
        ).first()

    if not payment:
        logger.warning(f"IPN: No payment found for tracking={order_tracking_id} ref={merchant_reference}")
        return {"status": "ignored", "reason": "payment not found"}

    logger.info(f"Found payment record: ID={payment.id}, current_status={payment.status}")

    # Update tracking ID if needed
    if order_tracking_id and not payment.order_tracking_id:
        logger.info(f"Updating payment with tracking ID: {order_tracking_id}")
        payment.order_tracking_id = order_tracking_id
        db.commit()

    # Query PesaPal for definitive status
    try:
        logger.info(f"Querying PesaPal for transaction status: {order_tracking_id}")
        pesapal_token = pesapal_get_token()
        status_data = pesapal_get_transaction_status(pesapal_token, order_tracking_id)
        
        payment_status_code = status_data.get("payment_status_description", "").upper()
        pesapal_status = status_data.get("status_code")
        
        logger.info(f"IPN status - payment_status_description: {payment_status_code}")
        logger.info(f"IPN status - status_code: {pesapal_status}")
        logger.info(f"IPN full response: {json.dumps(status_data, indent=2)}")

        if payment_status_code == "COMPLETED" or pesapal_status == 1:
            if payment.status != "COMPLETED":
                logger.info(f"IPN: Marking payment {payment.merchant_reference} as COMPLETED")
                payment.status = "COMPLETED"
                payment.updated_at = datetime.utcnow()
                db.commit()
                _enroll_user_in_course(db, payment.user_id, payment.course_id)
                logger.info(f"IPN: User {payment.user_id} enrolled in course {payment.course_id}")
            else:
                logger.info(f"IPN: Payment already marked as COMPLETED")
                
        elif payment_status_code in ("FAILED", "INVALID") or pesapal_status in (0, 2):
            if payment.status != "FAILED":
                logger.warning(f"IPN: Marking payment {payment.merchant_reference} as FAILED")
                payment.status = "FAILED"
                payment.updated_at = datetime.utcnow()
                db.commit()
                logger.info(f"IPN: Payment marked FAILED")
            else:
                logger.info(f"IPN: Payment already marked as FAILED")
        else:
            logger.info(f"IPN: Payment status unchanged: {payment_status_code}")

    except Exception as e:
        logger.error(f"IPN processing error: {e}", exc_info=True)

    # PesaPal expects a 200 OK with the orderNotificationType echoed back
    response_data = {
        "orderNotificationType": order_notification_type, 
        "orderTrackingId": order_tracking_id, 
        "orderMerchantReference": merchant_reference
    }
    logger.info(f"IPN response: {json.dumps(response_data)}")
    return response_data


@app.get("/payments/verify/{merchant_reference}", response_model=PaymentStatusResponse)
async def verify_payment(
    merchant_reference: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Allows the frontend to poll payment status after returning from PesaPal.
    If payment is COMPLETED, also triggers enrollment (safety net).
    """
    logger.info(f"=== PAYMENT VERIFICATION ===")
    logger.info(f"Merchant reference: {merchant_reference}, User: {current_user.id}")

    payment = db.query(PaymentModel).filter(
        PaymentModel.merchant_reference == merchant_reference,
        PaymentModel.user_id == current_user.id,
    ).first()

    if not payment:
        logger.error(f"Payment record not found for ref: {merchant_reference}")
        raise HTTPException(status_code=404, detail="Payment record not found.")

    logger.info(f"Payment found: status={payment.status}, tracking_id={payment.order_tracking_id}")

    # If still pending, re-query PesaPal for latest status
    if payment.status == "PENDING" and payment.order_tracking_id:
        logger.info(f"Payment still pending, querying PesaPal for status...")
        try:
            pesapal_token = pesapal_get_token()
            status_data = pesapal_get_transaction_status(pesapal_token, payment.order_tracking_id)
            
            payment_status_code = status_data.get("payment_status_description", "").upper()
            pesapal_status = status_data.get("status_code")
            
            logger.info(f"Verification status: {payment_status_code}, code={pesapal_status}")

            if payment_status_code == "COMPLETED" or pesapal_status == 1:
                logger.info(f"Payment verified as COMPLETED!")
                payment.status = "COMPLETED"
                payment.updated_at = datetime.utcnow()
                db.commit()
                _enroll_user_in_course(db, payment.user_id, payment.course_id)
            elif payment_status_code in ("FAILED", "INVALID") or pesapal_status in (0, 2):
                logger.warning(f"Payment verified as FAILED")
                payment.status = "FAILED"
                payment.updated_at = datetime.utcnow()
                db.commit()
            else:
                logger.info(f"Payment still pending according to PesaPal")
        except Exception as e:
            logger.error(f"Error re-querying PesaPal in verify endpoint: {e}", exc_info=True)

    logger.info(f"Final payment status: {payment.status}")

    return PaymentStatusResponse(
        merchant_reference=payment.merchant_reference,
        order_tracking_id=payment.order_tracking_id,
        status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        course_id=payment.course_id,
        created_at=payment.created_at,
    )


@app.get("/payments/my-payments")
async def get_my_payments(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all payment records for the current user."""
    logger.info(f"Fetching payments for user: {current_user.id}")
    
    payments = db.query(PaymentModel).filter(
        PaymentModel.user_id == current_user.id
    ).order_by(PaymentModel.created_at.desc()).all()
    
    logger.info(f"Found {len(payments)} payments")

    result = []
    for p in payments:
        course = db.query(CourseModel).filter(CourseModel.id == p.course_id).first()
        result.append({
            "id": p.id,
            "course_id": p.course_id,
            "course_title": course.title if course else "Unknown",
            "merchant_reference": p.merchant_reference,
            "order_tracking_id": p.order_tracking_id,
            "amount": p.amount,
            "currency": p.currency,
            "status": p.status,
            "created_at": p.created_at,
        })
    return result


# ---------------------------------------------------------------------------
# Profile Routes  ← NEW
# ---------------------------------------------------------------------------

@app.get("/users/me/profile", response_model=ProfileResponse)
async def get_my_profile(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return full profile for the currently authenticated user, including presigned avatar URL."""
    profile_image_url = None
    if current_user.profile_image_filename:
        profile_image_url = f"/b2-proxy/{current_user.profile_image_filename}"

    return ProfileResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_instructor=current_user.is_instructor,
        created_at=current_user.created_at,
        bio=current_user.bio,
        profile_image_url=profile_image_url,
        location=current_user.location,
        website=current_user.website,
        phone=current_user.phone,
    )


@app.put("/users/me/profile", response_model=ProfileResponse)
async def update_my_profile(
    full_name: Optional[str] = Form(None),
    bio: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    website: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    profile_image: Optional[UploadFile] = File(None),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update profile fields and/or avatar image for the current user."""
    user = db.query(UserModel).filter(UserModel.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if full_name is not None:
        user.full_name = full_name
    if bio is not None:
        user.bio = bio
    if location is not None:
        user.location = location
    if website is not None:
        user.website = website
    if phone is not None:
        user.phone = phone

    if profile_image:
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if profile_image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Invalid image type. Only JPEG, PNG, GIF, and WebP are allowed.",
            )
        # Delete old avatar from B2 if it exists
        if user.profile_image_filename:
            try:
                await delete_from_b2(user.profile_image_filename)
            except Exception as e:
                logger.warning(f"Could not delete old profile image: {e}")
        user.profile_image_filename = await upload_to_b2(profile_image, "avatars")
        logger.info(f"Profile image uploaded for user {user.id}: {user.profile_image_filename}")

    db.commit()
    db.refresh(user)

    profile_image_url = None
    if user.profile_image_filename:
        profile_image_url = f"/b2-proxy/{user.profile_image_filename}"

    logger.info(f"Profile updated for user {user.id}")
    return ProfileResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_instructor=user.is_instructor,
        created_at=user.created_at,
        bio=user.bio,
        profile_image_url=profile_image_url,
        location=user.location,
        website=user.website,
        phone=user.phone,
    )


@app.delete("/users/me/profile-image")
async def delete_profile_image(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove the current user's profile image."""
    user = db.query(UserModel).filter(UserModel.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.profile_image_filename:
        raise HTTPException(status_code=404, detail="No profile image to delete")

    try:
        await delete_from_b2(user.profile_image_filename)
    except Exception as e:
        logger.warning(f"Could not delete profile image from B2: {e}")

    user.profile_image_filename = None
    db.commit()
    return {"message": "Profile image removed successfully"}


# ---------------------------------------------------------------------------
# Course Image URL endpoints
# ---------------------------------------------------------------------------

@app.get("/courses/{course_id}/image-url", response_model=ImageUrlResponse)
async def get_course_image_url_endpoint(
    course_id: int,
    db: Session = Depends(get_db),
):
    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    key = _resolve_image_filename(course)
    if not key:
        raise HTTPException(status_code=404, detail="Course has no image")

    try:
        b2_client.head_object(Bucket=B2_BUCKET_NAME, Key=key)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        logger.warning(f"Image key '{key}' not found in B2: {error_code}")
        raise HTTPException(status_code=404, detail="Image file not found in storage")

    presigned_url = await generate_presigned_url(key, expiration=3600)
    if not presigned_url:
        raise HTTPException(status_code=500, detail="Could not generate image URL")

    return {"url": presigned_url, "expires_in": 3600}


@app.post("/courses/image-urls")
async def get_bulk_course_image_urls(
    course_ids: List[int],
    db: Session = Depends(get_db),
):
    result = {}
    courses = db.query(CourseModel).filter(CourseModel.id.in_(course_ids)).all()

    for course in courses:
        key = _resolve_image_filename(course)
        if not key:
            continue
        try:
            b2_client.head_object(Bucket=B2_BUCKET_NAME, Key=key)
            url = await generate_presigned_url(key, expiration=3600)
            if url:
                result[str(course.id)] = url
        except ClientError:
            logger.warning(f"Bulk image: key '{key}' not found for course {course.id}")
        except Exception as e:
            logger.error(f"Bulk image error for course {course.id}: {e}")

    return result


# ---------------------------------------------------------------------------
# Video Streaming
# ---------------------------------------------------------------------------

_VIDEO_CHUNK_SIZE = 512 * 1024  # 512 KB


@app.get("/stream/video/{filename:path}")
async def stream_video(
    filename: str,
    request: Request,
    token: str = Query(..., description="JWT token for video access"),
    db: Session = Depends(get_db),
):
    logger.info(f"Video streaming request for filename: {filename}")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        logger.error(f"JWT Decode Error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired video token")

    user_id = payload.get("user_id")
    lesson_id = payload.get("lesson_id")
    if not user_id or not lesson_id:
        raise HTTPException(status_code=401, detail="Invalid video token payload")

    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")

    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == user_id,
        EnrollmentModel.course_id == lesson.module.course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    if not lesson.video_filename:
        raise HTTPException(status_code=404, detail="No video for this lesson")

    object_key = lesson.video_filename
    logger.info(f"Streaming object key: {object_key}")

    try:
        head = b2_client.head_object(Bucket=B2_BUCKET_NAME, Key=object_key)
        file_size = head["ContentLength"]
        content_type = head.get("ContentType") or "video/mp4"
        logger.info(f"B2 object OK. size={file_size} type={content_type}")
    except ClientError as e:
        logger.error(f"B2 HEAD error: {e}")
        raise HTTPException(status_code=404, detail="Video file not found in storage")

    range_header = request.headers.get("Range")
    get_kwargs: dict = {"Bucket": B2_BUCKET_NAME, "Key": object_key}
    status_code = 200
    content_range = None

    if range_header:
        try:
            range_type, range_value = range_header.split("=", 1)
            start_str, end_str = range_value.split("-", 1)
            start = int(start_str) if start_str.strip() else 0
            end = int(end_str) if end_str.strip() else file_size - 1
            end = min(end, file_size - 1)

            if start > end or start >= file_size:
                raise HTTPException(status_code=416, detail="Range Not Satisfiable")

            get_kwargs["Range"] = f"bytes={start}-{end}"
            content_length = end - start + 1
            status_code = 206
            content_range = f"bytes {start}-{end}/{file_size}"
            logger.info(f"Range request: {get_kwargs['Range']} ({content_length} bytes)")
        except (ValueError, AttributeError):
            range_header = None

    if not range_header:
        start = 0
        end = file_size - 1
        content_length = file_size

    try:
        b2_response = b2_client.get_object(**get_kwargs)
    except ClientError as e:
        logger.error(f"B2 GET error: {e}")
        raise HTTPException(status_code=502, detail="Storage fetch error")

    def _stream_video_chunks(body):
        bytes_sent = 0
        try:
            with body as stream:
                while True:
                    chunk = stream.read(_VIDEO_CHUNK_SIZE)
                    if not chunk:
                        break
                    bytes_sent += len(chunk)
                    yield chunk
        except Exception as exc:
            logger.error(
                f"Stream error for '{object_key}' after {bytes_sent} bytes: {exc}"
            )

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "Range, Content-Type",
        "Access-Control-Expose-Headers": "Content-Range, Content-Length, Accept-Ranges",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": content_type,
        "X-Accel-Buffering": "no",
        "Cache-Control": "public, max-age=3600",
    }
    if content_range:
        headers["Content-Range"] = content_range

    return StreamingResponse(
        _stream_video_chunks(b2_response["Body"]),
        status_code=status_code,
        headers=headers,
        media_type=content_type,
    )


# ---------------------------------------------------------------------------
# Video Token
# ---------------------------------------------------------------------------

@app.get("/video-token/{lesson_id}", response_model=VideoTokenResponse)
async def get_video_token_by_path(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logger.info(f"Video token request for lesson {lesson_id} from user {current_user.id}")
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    expires_delta = timedelta(minutes=VIDEO_TOKEN_EXPIRE_MINUTES)
    token_data = {"sub": "video_access", "user_id": current_user.id, "lesson_id": lesson_id}
    access_token = create_video_token(token_data, expires_delta=expires_delta)
    logger.info(f"Video token generated for user {current_user.id}, lesson {lesson_id}")
    return {"token": access_token, "expires_at": datetime.utcnow() + expires_delta}


# ---------------------------------------------------------------------------
# Quiz Endpoints
# ---------------------------------------------------------------------------

@app.post("/lessons/{lesson_id}/generate-quiz", response_model=QuizResponse)
async def generate_quiz_for_lesson(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logger.info(f"Generating quiz for lesson {lesson_id} by user {current_user.id}")
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")

    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    if not lesson.content or lesson.content.strip() == "":
        raise HTTPException(status_code=400, detail="Lesson has no content to generate quiz from")

    if not lesson.has_quiz:
        raise HTTPException(status_code=400, detail="Quiz is not enabled for this lesson")

    quiz_id = str(uuid.uuid4())
    quiz_questions = await generate_quiz(lesson.content)
    cache_quiz_questions(current_user.id, lesson_id, quiz_questions, quiz_id)

    return {"questions": quiz_questions, "quiz_id": quiz_id}


@app.post("/lessons/{lesson_id}/submit-quiz", response_model=QuizResult)
async def submit_quiz(
    lesson_id: int,
    submission: QuizSubmission,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logger.info(f"Submitting quiz for lesson {lesson_id} by user {current_user.id}")
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")

    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    if not lesson.has_quiz:
        raise HTTPException(status_code=400, detail="Quiz is not enabled for this lesson")

    quiz_questions = get_cached_quiz_questions(current_user.id, lesson_id, submission.quiz_id)
    if not quiz_questions:
        raise HTTPException(
            status_code=400,
            detail="Quiz session expired or invalid. Please generate a new quiz.",
        )

    if len(submission.answers) != 5:
        raise HTTPException(status_code=400, detail="You must answer all 5 questions")

    for i, answer in enumerate(submission.answers):
        if not isinstance(answer, int) or answer < 0 or answer > 3:
            raise HTTPException(status_code=400, detail=f"Answer {i+1} must be between 0 and 3")

    score = 0
    correct_answers = []
    for i, question in enumerate(quiz_questions):
        correct_answer = question["correct_answer"]
        correct_answers.append(correct_answer)
        if i < len(submission.answers) and submission.answers[i] == correct_answer:
            score += 1

    passed = score >= 3

    quiz_attempt = QuizAttemptModel(
        user_id=current_user.id,
        lesson_id=lesson_id,
        questions=quiz_questions,
        user_answers=submission.answers,
        score=score,
        passed=passed,
    )
    db.add(quiz_attempt)
    db.commit()
    db.refresh(quiz_attempt)

    logger.info(f"Quiz submitted: score {score}/5, passed: {passed}")
    return {
        "score": score,
        "total": 5,
        "passed": passed,
        "correct_answers": correct_answers,
        "attempt_id": quiz_attempt.id,
        "questions": quiz_questions,
    }


@app.get("/lessons/{lesson_id}/quiz-attempts", response_model=List[QuizAttemptResponse])
async def get_quiz_attempts(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")

    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    attempts = db.query(QuizAttemptModel).filter(
        QuizAttemptModel.user_id == current_user.id,
        QuizAttemptModel.lesson_id == lesson_id,
    ).order_by(QuizAttemptModel.created_at.desc()).all()
    return attempts


# ---------------------------------------------------------------------------
# Certificate Endpoints
# ---------------------------------------------------------------------------

@app.get("/courses/{course_id}/certificate/eligibility", response_model=CertificateEligibilityResponse)
async def check_certificate_eligibility(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    existing_certificate = db.query(CertificateModel).filter(
        CertificateModel.user_id == current_user.id,
        CertificateModel.course_id == course_id,
    ).first()

    if existing_certificate:
        download_url = None
        if existing_certificate.pdf_filename:
            try:
                download_url = await generate_presigned_url(existing_certificate.pdf_filename)
            except Exception as e:
                logger.error(f"Error generating presigned URL: {e}")

        course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
        course_title = course.title if course else None

        return CertificateEligibilityResponse(
            eligible=True,
            completed_lessons=0,
            total_lessons=0,
            progress_percentage=100,
            message="You already have a certificate for this course",
            existing_certificate=CertificateResponse(
                id=existing_certificate.id,
                user_id=existing_certificate.user_id,
                course_id=existing_certificate.course_id,
                course_title=course_title,
                issued_at=existing_certificate.issued_at,
                certificate_hash=existing_certificate.certificate_hash,
                download_url=download_url,
            ),
        )

    is_completed, completed_lessons, total_lessons, progress_percentage = check_course_completion(
        db, current_user.id, course_id
    )

    if is_completed:
        return CertificateEligibilityResponse(
            eligible=True,
            completed_lessons=completed_lessons,
            total_lessons=total_lessons,
            progress_percentage=progress_percentage,
            message="Congratulations! You've completed all lessons and are eligible for a certificate.",
        )
    return CertificateEligibilityResponse(
        eligible=False,
        completed_lessons=completed_lessons,
        total_lessons=total_lessons,
        progress_percentage=progress_percentage,
        message=f"Complete all lessons to claim your certificate. You've completed {completed_lessons} of {total_lessons} lessons.",
    )


@app.post("/courses/{course_id}/certificate/claim", response_model=CertificateResponse)
async def claim_certificate(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    existing_certificate = db.query(CertificateModel).filter(
        CertificateModel.user_id == current_user.id,
        CertificateModel.course_id == course_id,
    ).first()
    if existing_certificate:
        raise HTTPException(status_code=400, detail="You already have a certificate for this course")

    is_completed, completed_lessons, total_lessons, _ = check_course_completion(
        db, current_user.id, course_id
    )
    if not is_completed:
        raise HTTPException(
            status_code=400,
            detail=f"You must complete all lessons to claim a certificate. You've completed {completed_lessons} of {total_lessons} lessons.",
        )

    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    certificate_hash = generate_certificate_hash(current_user.id, course_id)
    pdf_buffer = create_certificate_pdf(current_user, course, certificate_hash)
    pdf_filename = f"certificates/{certificate_hash}.pdf"
    await upload_certificate_to_b2(pdf_buffer, pdf_filename)

    certificate = CertificateModel(
        user_id=current_user.id,
        course_id=course_id,
        certificate_hash=certificate_hash,
        pdf_filename=pdf_filename,
    )
    db.add(certificate)
    db.commit()
    db.refresh(certificate)

    download_url = await generate_presigned_url(pdf_filename)
    logger.info(f"Certificate created for user {current_user.id} for course {course_id}")

    return CertificateResponse(
        id=certificate.id,
        user_id=certificate.user_id,
        course_id=certificate.course_id,
        course_title=course.title,
        issued_at=certificate.issued_at,
        certificate_hash=certificate.certificate_hash,
        download_url=download_url,
    )


@app.get("/certificates", response_model=List[CertificateResponse])
async def get_user_certificates(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    certificates = db.query(CertificateModel).filter(
        CertificateModel.user_id == current_user.id
    ).all()

    certificate_responses = []
    for certificate in certificates:
        course = db.query(CourseModel).filter(CourseModel.id == certificate.course_id).first()
        if not course:
            continue
        download_url = None
        if certificate.pdf_filename:
            try:
                download_url = await generate_presigned_url(certificate.pdf_filename)
            except Exception as e:
                logger.error(f"Error generating presigned URL for certificate {certificate.id}: {e}")

        certificate_responses.append(CertificateResponse(
            id=certificate.id,
            user_id=certificate.user_id,
            course_id=certificate.course_id,
            course_title=course.title,
            issued_at=certificate.issued_at,
            certificate_hash=certificate.certificate_hash,
            download_url=download_url,
        ))
    return certificate_responses


@app.get("/certificates/{certificate_hash}/verify")
async def verify_certificate(certificate_hash: str, db: Session = Depends(get_db)):
    certificate = db.query(CertificateModel).filter(
        CertificateModel.certificate_hash == certificate_hash
    ).first()
    if not certificate:
        raise HTTPException(status_code=404, detail="Certificate not found")

    user = db.query(UserModel).filter(UserModel.id == certificate.user_id).first()
    course = db.query(CourseModel).filter(CourseModel.id == certificate.course_id).first()
    if not user or not course:
        raise HTTPException(status_code=404, detail="Certificate details not found")

    return {
        "valid": True,
        "certificate_id": certificate.id,
        "user_name": user.full_name,
        "user_email": user.email,
        "course_title": course.title,
        "issued_at": certificate.issued_at,
        "certificate_hash": certificate.certificate_hash,
    }


@app.get("/certificates/{certificate_hash}/download")
async def download_certificate(
    certificate_hash: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    certificate = db.query(CertificateModel).filter(
        CertificateModel.certificate_hash == certificate_hash,
        CertificateModel.user_id == current_user.id,
    ).first()
    if not certificate:
        raise HTTPException(status_code=404, detail="Certificate not found")
    if not certificate.pdf_filename:
        raise HTTPException(status_code=404, detail="Certificate file not available")

    try:
        response = b2_client.get_object(Bucket=B2_BUCKET_NAME, Key=certificate.pdf_filename)
        file_size = response['ContentLength']
        content_type = response['ContentType'] or 'application/pdf'

        def generate_chunks():
            with response['Body'] as stream:
                while True:
                    chunk = stream.read(8192)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            generate_chunks(),
            media_type=content_type,
            headers={
                'Content-Disposition': f'attachment; filename="certificate-{certificate_hash}.pdf"',
                'Content-Length': str(file_size),
            },
        )
    except ClientError as e:
        logger.error(f"Error downloading certificate from B2: {e}")
        raise HTTPException(status_code=500, detail="Error downloading certificate")


# ---------------------------------------------------------------------------
# Auth Routes
# ---------------------------------------------------------------------------

@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    logger.info(f"Login attempt for user: {form_data.username}")
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
    logger.info(f"Successful login for user: {form_data.username}")
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/users/", response_model=User)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    logger.info(f"Creating new user: {user.email}")
    db_user = get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    db_user = UserModel(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        is_instructor=user.is_instructor,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info(f"User created successfully: {user.email}")
    return db_user


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user


# ---------------------------------------------------------------------------
# Course Routes
# ---------------------------------------------------------------------------

@app.post("/courses/", response_model=Course)
async def create_course(
    title: str = Form(...),
    description: str = Form(None),
    image_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logger.info(f"Creating course: {title} by user: {current_user.email}")
    if not current_user.is_instructor:
        raise HTTPException(status_code=403, detail="Only instructors can create courses")

    image_filename = None
    if image_file:
        allowed_image_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if image_file.content_type not in allowed_image_types:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPEG, PNG, GIF, and WebP images are allowed.",
            )
        image_filename = await upload_to_b2(image_file, "courses")
        logger.info(f"Course image uploaded to B2: {image_filename}")

    db_course = CourseModel(
        title=title,
        description=description,
        instructor_id=current_user.id,
        image_filename=image_filename,
    )
    db.add(db_course)
    db.commit()
    db.refresh(db_course)

    image_url = await get_course_image_url(db_course)
    return _build_course_response(db_course, current_user.full_name, image_url)


@app.get("/courses/", response_model=List[Course])
async def read_courses(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    logger.info(f"Fetching courses, skip: {skip}, limit: {limit}")
    courses = db.query(CourseModel).filter(CourseModel.is_published == True).offset(skip).limit(limit).all()

    course_list = []
    for course in courses:
        image_url = await get_course_image_url(course)
        instructor = db.query(UserModel).filter(UserModel.id == course.instructor_id).first()
        instructor_name = instructor.full_name if instructor else "Unknown Instructor"
        course_list.append(_build_course_response(course, instructor_name, image_url))

    logger.info(f"Returning {len(course_list)} courses")
    return course_list


@app.get("/courses/{course_id}", response_model=Course)
async def read_course(course_id: int, db: Session = Depends(get_db)):
    logger.info(f"Fetching course: {course_id}")
    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    if course is None:
        raise HTTPException(status_code=404, detail="Course not found")

    image_url = await get_course_image_url(course)
    instructor = db.query(UserModel).filter(UserModel.id == course.instructor_id).first()
    instructor_name = instructor.full_name if instructor else "Unknown Instructor"
    return _build_course_response(course, instructor_name, image_url)


@app.put("/courses/{course_id}", response_model=Course)
async def update_course(
    course_id: int,
    title: str = Form(None),
    description: str = Form(None),
    image_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logger.info(f"Updating course: {course_id} by user: {current_user.email}")
    if not current_user.is_instructor:
        raise HTTPException(status_code=403, detail="Only instructors can update courses")

    course = db.query(CourseModel).filter(
        CourseModel.id == course_id, CourseModel.instructor_id == current_user.id
    ).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found or you don't have permission")

    if title is not None:
        course.title = title
    if description is not None:
        course.description = description

    if image_file:
        allowed_image_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if image_file.content_type not in allowed_image_types:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPEG, PNG, GIF, and WebP images are allowed.",
            )
        if course.image_filename:
            await delete_from_b2(course.image_filename)
        course.image_filename = await upload_to_b2(image_file, "courses")
        logger.info(f"Updated course image uploaded to B2: {course.image_filename}")

    db.commit()
    db.refresh(course)

    image_url = await get_course_image_url(course)
    instructor = db.query(UserModel).filter(UserModel.id == course.instructor_id).first()
    instructor_name = instructor.full_name if instructor else "Unknown Instructor"
    logger.info(f"Course updated successfully: {course_id}")
    return _build_course_response(course, instructor_name, image_url)


@app.put("/courses/{course_id}/publish/")
def publish_course(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.is_instructor:
        raise HTTPException(status_code=403, detail="Only instructors can publish courses")
    course = db.query(CourseModel).filter(
        CourseModel.id == course_id, CourseModel.instructor_id == current_user.id
    ).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found or you don't have permission")
    course.is_published = True
    db.commit()
    return {"message": "Course published successfully"}


# ---------------------------------------------------------------------------
# Module Routes
# ---------------------------------------------------------------------------

@app.post("/courses/{course_id}/modules/", response_model=Module)
def create_module(
    course_id: int,
    module: ModuleCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.is_instructor:
        raise HTTPException(status_code=403, detail="Only instructors can create modules")
    course = db.query(CourseModel).filter(
        CourseModel.id == course_id, CourseModel.instructor_id == current_user.id
    ).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found or you don't have permission")
    db_module = ModuleModel(
        title=module.title, description=module.description, course_id=course_id, order=module.order
    )
    db.add(db_module)
    db.commit()
    db.refresh(db_module)
    return db_module


@app.get("/courses/{course_id}/modules/", response_model=List[Module])
def get_course_modules(course_id: int, db: Session = Depends(get_db)):
    return db.query(ModuleModel).filter(ModuleModel.course_id == course_id).order_by(ModuleModel.order).all()


@app.put("/modules/{module_id}", response_model=Module)
def update_module(
    module_id: int,
    module_data: ModuleBase,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.is_instructor:
        raise HTTPException(status_code=403, detail="Only instructors can update modules")
    module = db.query(ModuleModel).join(CourseModel).filter(
        ModuleModel.id == module_id, CourseModel.instructor_id == current_user.id
    ).first()
    if not module:
        raise HTTPException(status_code=404, detail="Module not found or you don't have permission")
    module.title = module_data.title
    module.description = module_data.description
    module.order = module_data.order
    db.commit()
    db.refresh(module)
    return module


@app.delete("/modules/{module_id}")
def delete_module(
    module_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.is_instructor:
        raise HTTPException(status_code=403, detail="Only instructors can delete modules")
    module = db.query(ModuleModel).join(CourseModel).filter(
        ModuleModel.id == module_id, CourseModel.instructor_id == current_user.id
    ).first()
    if not module:
        raise HTTPException(status_code=404, detail="Module not found or you don't have permission")
    for lesson in module.lessons:
        if lesson.video_filename:
            asyncio.run(delete_from_b2(lesson.video_filename))
    db.delete(module)
    db.commit()
    return {"message": "Module deleted successfully"}


# ---------------------------------------------------------------------------
# Lesson Routes
# ---------------------------------------------------------------------------

@app.post("/modules/{module_id}/lessons/", response_model=LessonResponse)
async def create_lesson(
    module_id: int,
    title: str = Form(...),
    content: str = Form(None),
    order: int = Form(1),
    has_quiz: bool = Form(True),
    video_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.is_instructor:
        raise HTTPException(status_code=403, detail="Only instructors can create lessons")
    module = db.query(ModuleModel).join(CourseModel).filter(
        ModuleModel.id == module_id, CourseModel.instructor_id == current_user.id
    ).first()
    if not module:
        raise HTTPException(status_code=404, detail="Module not found or you don't have permission")

    video_filename = None
    if video_file:
        allowed_video_types = ["video/mp4", "video/mov", "video/avi", "video/webm", "video/quicktime"]
        if video_file.content_type not in allowed_video_types:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only MP4, MOV, AVI, and WebM files are allowed.",
            )
        video_filename = await upload_to_b2(video_file, "lessons")
        logger.info(f"Video uploaded to B2: {video_filename}")

    db_lesson = LessonModel(
        title=title, content=content, module_id=module_id,
        order=order, has_quiz=has_quiz, video_filename=video_filename,
    )
    db.add(db_lesson)
    db.commit()
    db.refresh(db_lesson)

    video_url = f"/stream/video/{video_filename}" if video_filename else None
    return {
        "id": db_lesson.id, "title": db_lesson.title, "content": db_lesson.content,
        "order": db_lesson.order, "module_id": db_lesson.module_id, "has_quiz": db_lesson.has_quiz,
        "video_url": video_url, "video_filename": db_lesson.video_filename,
    }


@app.get("/modules/{module_id}/lessons/", response_model=List[LessonResponse])
async def get_module_lessons(
    module_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lessons = db.query(LessonModel).filter(LessonModel.module_id == module_id).order_by(LessonModel.order).all()
    module = db.query(ModuleModel).filter(ModuleModel.id == module_id).first()
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == module.course_id,
    ).first()

    response_lessons = []
    for lesson in lessons:
        completed = False
        if enrollment:
            progress = db.query(ProgressModel).filter(
                ProgressModel.enrollment_id == enrollment.id,
                ProgressModel.lesson_id == lesson.id,
                ProgressModel.completed == True,
            ).first()
            completed = progress is not None

        video_url = f"/stream/video/{lesson.video_filename}" if lesson.video_filename else None
        response_lessons.append({
            "id": lesson.id, "title": lesson.title, "content": lesson.content,
            "order": lesson.order, "module_id": lesson.module_id, "has_quiz": lesson.has_quiz,
            "video_url": video_url, "video_filename": lesson.video_filename, "completed": completed,
        })
    return response_lessons


@app.get("/lessons/{lesson_id}", response_model=LessonResponse)
async def get_lesson(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")

    module = db.query(ModuleModel).filter(ModuleModel.id == lesson.module_id).first()
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == module.course_id,
    ).first()

    completed = False
    if enrollment:
        progress = db.query(ProgressModel).filter(
            ProgressModel.enrollment_id == enrollment.id,
            ProgressModel.lesson_id == lesson_id,
            ProgressModel.completed == True,
        ).first()
        completed = progress is not None

    video_url = f"/stream/video/{lesson.video_filename}" if lesson.video_filename else None
    return {
        "id": lesson.id, "title": lesson.title, "content": lesson.content,
        "order": lesson.order, "module_id": lesson.module_id, "has_quiz": lesson.has_quiz,
        "video_url": video_url, "video_filename": lesson.video_filename, "completed": completed,
    }


@app.put("/lessons/{lesson_id}", response_model=LessonResponse)
async def update_lesson(
    lesson_id: int,
    title: str = Form(None),
    content: str = Form(None),
    order: int = Form(None),
    has_quiz: bool = Form(None),
    video_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.is_instructor:
        raise HTTPException(status_code=403, detail="Only instructors can update lessons")
    lesson = db.query(LessonModel).join(ModuleModel).join(CourseModel).filter(
        LessonModel.id == lesson_id, CourseModel.instructor_id == current_user.id
    ).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found or you don't have permission")

    if title is not None:
        lesson.title = title
    if content is not None:
        lesson.content = content
    if order is not None:
        lesson.order = order
    if has_quiz is not None:
        lesson.has_quiz = has_quiz

    if video_file:
        allowed_video_types = ["video/mp4", "video/mov", "video/avi", "video/webm", "video/quicktime"]
        if video_file.content_type not in allowed_video_types:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only MP4, MOV, AVI, and WebM files are allowed.",
            )
        if lesson.video_filename:
            await delete_from_b2(lesson.video_filename)
        lesson.video_filename = await upload_to_b2(video_file, "lessons")

    db.commit()
    db.refresh(lesson)

    video_url = f"/stream/video/{lesson.video_filename}" if lesson.video_filename else None
    return {
        "id": lesson.id, "title": lesson.title, "content": lesson.content,
        "order": lesson.order, "module_id": lesson.module_id, "has_quiz": lesson.has_quiz,
        "video_url": video_url, "video_filename": lesson.video_filename,
    }


@app.delete("/lessons/{lesson_id}")
async def delete_lesson(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.is_instructor:
        raise HTTPException(status_code=403, detail="Only instructors can delete lessons")
    lesson = db.query(LessonModel).join(ModuleModel).join(CourseModel).filter(
        LessonModel.id == lesson_id, CourseModel.instructor_id == current_user.id
    ).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found or you don't have permission")
    if lesson.video_filename:
        await delete_from_b2(lesson.video_filename)
    db.delete(lesson)
    db.commit()
    return {"message": "Lesson deleted successfully"}


# ---------------------------------------------------------------------------
# Enrollment Routes (kept for instructor use / direct enrollment)
# ---------------------------------------------------------------------------

@app.post("/enroll/{course_id}")
def enroll_in_course(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Direct enrollment (for instructors or free overrides). Regular users should pay via /payments/initiate/."""
    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    existing_enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == course_id,
    ).first()
    if existing_enrollment:
        raise HTTPException(status_code=400, detail="Already enrolled in this course")
    enrollment = EnrollmentModel(user_id=current_user.id, course_id=course_id)
    db.add(enrollment)
    db.commit()
    return {"message": "Successfully enrolled in course"}


@app.get("/my-courses/", response_model=List[Course])
async def get_my_courses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    enrollments = db.query(EnrollmentModel).filter(EnrollmentModel.user_id == current_user.id).all()
    course_ids = [e.course_id for e in enrollments]
    courses = db.query(CourseModel).filter(CourseModel.id.in_(course_ids)).all()

    course_list = []
    for course in courses:
        image_url = await get_course_image_url(course)
        instructor = db.query(UserModel).filter(UserModel.id == course.instructor_id).first()
        instructor_name = instructor.full_name if instructor else "Unknown Instructor"
        course_list.append(_build_course_response(course, instructor_name, image_url))
    return course_list


@app.get("/instructor/courses/", response_model=List[Course])
async def get_instructor_courses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.is_instructor:
        raise HTTPException(status_code=403, detail="Only instructors can access this endpoint")
    courses = db.query(CourseModel).filter(CourseModel.instructor_id == current_user.id).all()

    course_list = []
    for course in courses:
        image_url = await get_course_image_url(course)
        course_list.append(_build_course_response(course, current_user.full_name, image_url))
    return course_list


@app.get("/enrollment/check/{course_id}")
def check_enrollment_status(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == course_id,
    ).first()
    return {"is_enrolled": enrollment is not None}


# ---------------------------------------------------------------------------
# Progress Routes
# ---------------------------------------------------------------------------

@app.post("/progress/lesson/{lesson_id}/complete")
def mark_lesson_complete(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")

    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=404, detail="Not enrolled in this course")

    progress = db.query(ProgressModel).filter(
        ProgressModel.enrollment_id == enrollment.id,
        ProgressModel.lesson_id == lesson_id,
    ).first()

    if progress:
        if progress.completed:
            return {"message": "Lesson already completed"}
        progress.completed = True
        progress.completed_at = datetime.utcnow()
        db.commit()
        return {"message": "Lesson marked as complete"}

    progress = ProgressModel(
        enrollment_id=enrollment.id,
        lesson_id=lesson_id,
        completed=True,
        completed_at=datetime.utcnow(),
    )
    db.add(progress)
    db.commit()
    return {"message": "Lesson marked as complete"}


@app.get("/progress/course/{course_id}")
def get_course_progress(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=404, detail="Not enrolled in this course")

    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    total_lessons = 0
    completed_lessons = 0
    for module in course.modules:
        total_lessons += len(module.lessons)
        for lesson in module.lessons:
            progress = db.query(ProgressModel).filter(
                ProgressModel.enrollment_id == enrollment.id,
                ProgressModel.lesson_id == lesson.id,
                ProgressModel.completed == True,
            ).first()
            if progress:
                completed_lessons += 1

    progress_percentage = (completed_lessons / total_lessons * 100) if total_lessons > 0 else 0
    return {
        "total_lessons": total_lessons,
        "completed_lessons": completed_lessons,
        "progress_percentage": progress_percentage,
    }


@app.get("/lessons/{lesson_id}/access")
def check_lesson_access(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")

    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    can_access = check_previous_lessons_completed(db, enrollment.id, lesson)
    current_progress = db.query(ProgressModel).filter(
        ProgressModel.enrollment_id == enrollment.id,
        ProgressModel.lesson_id == lesson_id,
    ).first()
    current_lesson_completed = current_progress.completed if current_progress else False

    return {
        "can_access": can_access,
        "current_lesson_completed": current_lesson_completed,
        "message": "Access granted" if can_access else "Complete previous lessons first",
    }


# ---------------------------------------------------------------------------
# Video Token (lesson-scoped path)
# ---------------------------------------------------------------------------

@app.get("/lessons/{lesson_id}/video-token", response_model=VideoTokenResponse)
async def get_lesson_video_token(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")

    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    expires_delta = timedelta(minutes=VIDEO_TOKEN_EXPIRE_MINUTES)
    token_data = {"sub": "video_access", "user_id": current_user.id, "lesson_id": lesson_id}
    access_token = create_video_token(token_data, expires_delta=expires_delta)
    return {"token": access_token, "expires_at": datetime.utcnow() + expires_delta}


# ---------------------------------------------------------------------------
# Performance / Scoring Routes
# ---------------------------------------------------------------------------

@app.get("/modules/{module_id}/total-score")
def get_module_total_score(
    module_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    module = db.query(ModuleModel).filter(ModuleModel.id == module_id).first()
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == module.course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    lessons_with_quizzes = db.query(LessonModel).filter(
        LessonModel.module_id == module_id, LessonModel.has_quiz == True
    ).all()

    total_score = 0
    total_possible = 0
    quiz_results = []

    for lesson in lessons_with_quizzes:
        best_attempt = db.query(QuizAttemptModel).filter(
            QuizAttemptModel.user_id == current_user.id,
            QuizAttemptModel.lesson_id == lesson.id,
        ).order_by(QuizAttemptModel.score.desc()).first()

        if best_attempt:
            total_score += best_attempt.score
            total_possible += 5
            quiz_results.append({
                "lesson_id": lesson.id,
                "lesson_title": lesson.title,
                "score": best_attempt.score,
                "passed": best_attempt.passed,
                "attempt_date": best_attempt.created_at,
            })

    percentage = (total_score / total_possible * 100) if total_possible > 0 else 0
    return {
        "module_id": module_id,
        "module_title": module.title,
        "total_score": total_score,
        "total_possible": total_possible,
        "percentage": percentage,
        "quiz_results": quiz_results,
    }


@app.get("/performance/course/{course_id}")
def get_course_performance(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    progress_data = db.query(
        ProgressModel.lesson_id, ProgressModel.completed, ProgressModel.completed_at
    ).filter(ProgressModel.enrollment_id == enrollment.id).all()

    quiz_attempts = db.query(QuizAttemptModel).join(LessonModel).filter(
        QuizAttemptModel.user_id == current_user.id,
        LessonModel.module_id.in_([m.id for m in course.modules]),
    ).all()

    module_scores = []
    for module in course.modules:
        module_quiz_score = 0
        module_total_possible = 0
        for lesson in module.lessons:
            if lesson.has_quiz:
                best_attempt = None
                for attempt in quiz_attempts:
                    if attempt.lesson_id == lesson.id:
                        if not best_attempt or attempt.score > best_attempt.score:
                            best_attempt = attempt
                if best_attempt:
                    module_quiz_score += best_attempt.score
                module_total_possible += 5

        module_percentage = (module_quiz_score / module_total_possible * 100) if module_total_possible > 0 else 0
        module_scores.append({
            "module_id": module.id,
            "module_title": module.title,
            "quiz_score": module_quiz_score,
            "total_possible": module_total_possible,
            "percentage": module_percentage,
        })

    total_lessons = sum(len(module.lessons) for module in course.modules)
    completed_lessons = len([p for p in progress_data if p.completed])
    progress_percentage = (completed_lessons / total_lessons * 100) if total_lessons > 0 else 0

    total_quiz_score = sum(ms["quiz_score"] for ms in module_scores)
    total_quiz_possible = sum(ms["total_possible"] for ms in module_scores)
    overall_quiz_percentage = (total_quiz_score / total_quiz_possible * 100) if total_quiz_possible > 0 else 0

    return {
        "course_id": course_id,
        "course_title": course.title,
        "overall_progress": progress_percentage,
        "completed_lessons": completed_lessons,
        "total_lessons": total_lessons,
        "overall_quiz_score": total_quiz_score,
        "total_quiz_possible": total_quiz_possible,
        "overall_quiz_percentage": overall_quiz_percentage,
        "module_scores": module_scores,
        "enrollment_date": enrollment.enrolled_at,
    }


# ---------------------------------------------------------------------------
# PDF Download
# ---------------------------------------------------------------------------

@app.get("/lessons/{lesson_id}/download-pdf")
async def download_lesson_pdf(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")

    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{lesson.title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .content {{ margin-top: 20px; }}
            .footer {{ margin-top: 40px; text-align: center; color: #7f8c8d; font-size: 12px; }}
        </style>
    </head>
    <body>
        <h1>{lesson.title}</h1>
        <div class="content">
            {lesson.content or 'No content available for this lesson.'}
        </div>
        <div class="footer">
            Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")} | eLearning Platform
        </div>
    </body>
    </html>
    """

    try:
        import subprocess
        try:
            subprocess.run(['wkhtmltopdf', '--version'], capture_output=True, check=True)
            wkhtmltopdf_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            wkhtmltopdf_available = False

        if wkhtmltopdf_available:
            config = pdfkit.configuration(wkhtmltopdf='/usr/bin/wkhtmltopdf')
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                pdfkit.from_string(html_content, tmp_file.name, configuration=config)
                with open(tmp_file.name, 'rb') as f:
                    pdf_data = f.read()
                os.unlink(tmp_file.name)
        else:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            title_style = ParagraphStyle(
                'CustomTitle', parent=styles['Title'], fontSize=24,
                spaceAfter=30, textColor=colors.HexColor('#2C3E50'),
            )
            elements.append(Paragraph(lesson.title, title_style))
            if lesson.content:
                content_style = ParagraphStyle(
                    'CustomNormal', parent=styles['Normal'], fontSize=12, spaceAfter=12
                )
                elements.append(Paragraph(lesson.content, content_style))
            else:
                elements.append(Paragraph("No content available for this lesson.", styles['Normal']))
            doc.build(elements)
            pdf_data = buffer.getvalue()
            buffer.close()

        return Response(
            content=pdf_data,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={lesson.title.replace(' ', '_')}_notes.pdf",
                "Content-Length": str(len(pdf_data)),
            },
        )
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate PDF")


# ---------------------------------------------------------------------------
# Quiz Attempt Details
# ---------------------------------------------------------------------------

@app.get("/quiz-attempts/{attempt_id}", response_model=QuizAttemptDetailResponse)
async def get_quiz_attempt_details(
    attempt_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    quiz_attempt = db.query(QuizAttemptModel).filter(
        QuizAttemptModel.id == attempt_id,
        QuizAttemptModel.user_id == current_user.id,
    ).first()
    if not quiz_attempt:
        raise HTTPException(status_code=404, detail="Quiz attempt not found")

    questions_with_answers = []
    for i, question_data in enumerate(quiz_attempt.questions):
        user_answer = quiz_attempt.user_answers[i] if i < len(quiz_attempt.user_answers) else -1
        is_correct = user_answer == question_data["correct_answer"]
        questions_with_answers.append({
            "question": question_data["question"],
            "options": question_data["options"],
            "user_answer": user_answer,
            "correct_answer": question_data["correct_answer"],
            "is_correct": is_correct,
        })

    return {
        "id": quiz_attempt.id,
        "user_id": quiz_attempt.user_id,
        "lesson_id": quiz_attempt.lesson_id,
        "score": quiz_attempt.score,
        "total": 5,
        "passed": quiz_attempt.passed,
        "created_at": quiz_attempt.created_at,
        "questions": questions_with_answers,
    }




# ─────────────────────────────────────────────────────────────────────────────
# ADD THIS ENDPOINT anywhere after the quiz endpoints in main.py
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/lessons/{lesson_id}/ai-consult", response_model=AIConsultResponse)
async def ai_consult(
    lesson_id: int,
    request: AIConsultRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Let an enrolled student ask the AI tutor a question about a specific lesson.
    The lesson content is injected as context so the AI can give deep, relevant answers.
    Chat history is NOT stored — each request is stateless.
    """
    # ── Auth checks ──────────────────────────────────────────────────────────
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")

    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id,
    ).first()
    if not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    # ── Validate question ────────────────────────────────────────────────────
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question is too long (max 1000 characters)")

    # ── Build prompt ─────────────────────────────────────────────────────────
    lesson_context = lesson.content or "No written content available for this lesson."

    system_prompt = f"""You are an expert AI tutor for an online learning platform called ScienceTech Academy.
A student is asking a question about a specific lesson. Your role is to:
- Explain concepts clearly and in depth
- Use examples, analogies, and step-by-step breakdowns where helpful
- Stay focused on the lesson topic but connect to related concepts when useful
- Be encouraging and supportive
- Keep answers concise yet thorough (aim for 150–400 words unless the topic demands more)

LESSON TITLE: {lesson.title}

LESSON CONTENT (use this as your primary knowledge source):
\"\"\"
{lesson_context[:4000]}
\"\"\"

Answer the student's question below based on the lesson content above.
If the question goes beyond the lesson, you may draw on your broader knowledge
while noting that it extends the lesson material."""

    # ── Call Gemini ──────────────────────────────────────────────────────────
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY not set — cannot serve AI consultation")
        raise HTTPException(
            status_code=503,
            detail="AI consultation is temporarily unavailable. Please contact support.",
        )

    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt,
        )
        response = model.generate_content(question)
        answer = response.text.strip()

        if not answer:
            raise ValueError("Empty response from AI model")

        logger.info(f"AI consult: user={current_user.id} lesson={lesson_id} q_len={len(question)}")
        return AIConsultResponse(answer=answer)

    except Exception as e:
        logger.error(f"AI consultation error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get AI response. Please try again.",
        )
# ---------------------------------------------------------------------------
# File Upload
# ---------------------------------------------------------------------------

@app.post("/upload/course-image/")
async def upload_course_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    if not current_user.is_instructor:
        raise HTTPException(status_code=403, detail="Only instructors can upload images")
    filename = await upload_to_b2(file, "courses")
    return {"filename": filename, "url": f"/b2-proxy/{filename}"}


# ---------------------------------------------------------------------------
# B2 Proxy — TRUE streaming proxy
# ---------------------------------------------------------------------------

@app.get("/b2-proxy/{filename:path}")
async def b2_proxy(filename: str, request: Request):
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    clean_filename = filename.split("?")[0].strip("/")
    if not clean_filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    try:
        head = b2_client.head_object(Bucket=B2_BUCKET_NAME, Key=clean_filename)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ("404", "NoSuchKey", "403", "AccessDenied"):
            logger.warning(f"B2 proxy: object not found/accessible: {clean_filename} ({error_code})")
            raise HTTPException(status_code=404, detail="File not found")
        logger.error(f"B2 HEAD error for '{clean_filename}': {error_code}")
        raise HTTPException(status_code=502, detail="Storage error")

    file_size = head["ContentLength"]
    content_type = head.get("ContentType") or "application/octet-stream"
    etag = head.get("ETag", "").strip('"')

    range_header = request.headers.get("Range")
    get_kwargs: dict = {"Bucket": B2_BUCKET_NAME, "Key": clean_filename}

    if range_header:
        try:
            range_type, range_value = range_header.split("=", 1)
            start_str, end_str = range_value.split("-", 1)
            start = int(start_str) if start_str.strip() else 0
            end = int(end_str) if end_str.strip() else file_size - 1
            end = min(end, file_size - 1)
            if start > end or start >= file_size:
                raise HTTPException(status_code=416, detail="Range Not Satisfiable")
            get_kwargs["Range"] = f"bytes={start}-{end}"
            content_length = end - start + 1
            status_code = 206
            content_range = f"bytes {start}-{end}/{file_size}"
        except (ValueError, AttributeError):
            range_header = None

    if not range_header:
        start = 0
        end = file_size - 1
        content_length = file_size
        status_code = 200
        content_range = None

    try:
        b2_response = b2_client.get_object(**get_kwargs)
    except ClientError as e:
        logger.error(f"B2 GET error for '{clean_filename}': {e}")
        raise HTTPException(status_code=502, detail="Storage fetch error")

    def _stream_body(body, chunk_size: int = 65536):
        try:
            with body as stream:
                while True:
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception as exc:
            logger.error(f"B2 proxy stream error for '{clean_filename}': {exc}")

    headers = {
        "Content-Type": content_type,
        "Content-Length": str(content_length),
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "Range, Content-Type",
        "Access-Control-Expose-Headers": "Content-Range, Content-Length, Accept-Ranges",
        "X-Accel-Buffering": "no",
    }
    if etag:
        headers["ETag"] = f'"{etag}"'
    if content_range:
        headers["Content-Range"] = content_range

    return StreamingResponse(
        _stream_body(b2_response["Body"]),
        status_code=status_code,
        headers=headers,
        media_type=content_type,
    )


@app.options("/b2-proxy/{filename:path}")
async def b2_proxy_options(filename: str):
    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Range, Content-Type, Authorization",
            "Access-Control-Max-Age": "86400",
        },
    )


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

@app.post("/migrate")
def run_migrations(db: Session = Depends(get_db)):
    try:
        Base.metadata.create_all(bind=engine)
        quiz_migration_result = migrate_has_quiz_default(db)
        return {"message": "Database schema is up to date", "quiz_migration": quiz_migration_result}
    except Exception as e:
        logger.error(f"Migration error: {e}")
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")


# ---------------------------------------------------------------------------
# Debug Endpoints
# ---------------------------------------------------------------------------

@app.get("/debug/video/{lesson_id}")
async def debug_video_access(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        return {"error": "Lesson not found"}

    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id,
    ).first()

    b2_info = {}
    if lesson.video_filename:
        try:
            head_response = b2_client.head_object(Bucket=B2_BUCKET_NAME, Key=lesson.video_filename)
            b2_info = {
                "exists": True,
                "size": head_response['ContentLength'],
                "content_type": head_response['ContentType'],
                "last_modified": head_response['LastModified'],
            }
        except ClientError as e:
            b2_info = {"exists": False, "error": str(e)}

    return {
        "lesson_id": lesson_id,
        "lesson_title": lesson.title,
        "video_filename": lesson.video_filename,
        "enrolled": enrollment is not None,
        "b2_info": b2_info,
        "video_token_url": f"/video-token/{lesson_id}",
    }


@app.get("/debug/course-image/{course_id}")
async def debug_course_image(course_id: int, db: Session = Depends(get_db)):
    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    if not course:
        return {"error": "Course not found"}

    resolved_key = _resolve_image_filename(course)

    b2_info = {}
    if resolved_key:
        try:
            head_response = b2_client.head_object(Bucket=B2_BUCKET_NAME, Key=resolved_key)
            b2_info = {
                "exists": True,
                "size": head_response['ContentLength'],
                "content_type": head_response['ContentType'],
                "last_modified": head_response['LastModified'],
                "resolved_key": resolved_key,
            }
        except ClientError as e:
            b2_info = {"exists": False, "error": str(e), "resolved_key": resolved_key}

    presigned_url = await generate_presigned_url(resolved_key) if resolved_key else None

    return {
        "course_id": course_id,
        "course_title": course.title,
        "has_image_filename": bool(course.image_filename),
        "image_url_raw": course.image_url,
        "image_filename_raw": course.image_filename,
        "resolved_key": resolved_key,
        "b2_info": b2_info,
        "b2_proxy_url": f"/b2-proxy/{resolved_key}" if resolved_key else None,
        "direct_image_url_endpoint": f"/courses/{course_id}/image-url",
        "presigned_url_preview": presigned_url,
    }


@app.get("/debug/pesapal")
async def debug_pesapal():
    """Test PesaPal authentication and IPN registration. Remove in production."""
    logger.info("=== PESAPAL DEBUG ENDPOINT CALLED ===")
    try:
        token = pesapal_get_token()
        ipn_id = pesapal_register_ipn(token)
        return {
            "status": "ok",
            "token_cached": bool(token),
            "ipn_id": ipn_id,
            "base_url": PESAPAL_BASE_URL,
        }
    except HTTPException as e:
        logger.error(f"PesaPal debug error: {e.detail}")
        return {"status": "error", "detail": e.detail}


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("Running startup migrations...")
    db = SessionLocal()
    try:
        Base.metadata.create_all(bind=engine)
        migrate_has_quiz_default(db)
        # Add missing columns if they don't exist
        with engine.begin() as conn:
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS bio TEXT;
            """))
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS profile_image_filename VARCHAR(255);
            """))
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS location VARCHAR(255);
            """))
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS website VARCHAR(255);
            """))
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS phone VARCHAR(50);
            """))
        logger.info("Startup migrations completed successfully")
        # Pre-register PesaPal IPN on startup so it's ready for first payment
        try:
            token = pesapal_get_token()
            ipn_id = pesapal_register_ipn(token)
            logger.info(f"PesaPal IPN pre-registered on startup. ipn_id={ipn_id}")
        except Exception as pe:
            logger.warning(f"PesaPal IPN pre-registration failed on startup (will retry on first payment): {pe}")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    Base.metadata.create_all(bind=engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)
