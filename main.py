from fastapi import FastAPI, Depends, HTTPException, status, Form, UploadFile, File, BackgroundTasks, Request, Query, Header, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON, func, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime, timedelta
from typing import Optional, List, Union, Dict, Set
from jose import JWTError, jwt
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Set logging
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
PESAPAL_BASE_URL = os.getenv("PESAPAL_BASE_URL", "https://pay.pesapal.com/v3")
PESAPAL_IPN_URL = os.getenv("PESAPAL_IPN_URL", "https://elearning-1-r5di.onrender.com/payments/ipn")
PESAPAL_CALLBACK_URL = os.getenv("PESAPAL_CALLBACK_URL", "https://online-coderise.vercel.app/payment/callback")
COURSE_PRICE_UGX = 25000

# Platform branding
PLATFORM_NAME = "ScienceTech Academy"
PLATFORM_DOMAIN = "online-coderise.vercel.app"
PLATFORM_TAGLINE = "Empowering Learners Worldwide"

_pesapal_token_cache: dict = {}
_pesapal_ipn_id: Optional[str] = None

# ---------------------------------------------------------------------------
# Email Configuration
# ---------------------------------------------------------------------------
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER", "sciencetechacademy2026@gmail.com")        # e.g. yourplatform@gmail.com
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD", "htcr wuba yynf hokh") # App password for Gmail
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "ScienceTech Academy")
EMAIL_ENABLED = bool(EMAIL_HOST_USER and EMAIL_HOST_PASSWORD)

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    redis_client.ping()
    logger.info("Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
    redis_client = None

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
    sent_messages = relationship("MessageModel", foreign_keys="MessageModel.sender_id", back_populates="sender")
    received_messages = relationship("MessageModel", foreign_keys="MessageModel.recipient_id", back_populates="recipient")


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
    group_messages = relationship("GroupMessageModel", back_populates="course")


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
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    course_id = Column(Integer, ForeignKey("courses.id"))
    merchant_reference = Column(String, unique=True, index=True)
    order_tracking_id = Column(String, nullable=True, index=True)
    amount = Column(Integer, default=COURSE_PRICE_UGX)
    currency = Column(String, default="UGX")
    status = Column(String, default="PENDING")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("UserModel", back_populates="payments")
    course = relationship("CourseModel", back_populates="payments")


# ---------------------------------------------------------------------------
# Messaging Models
# ---------------------------------------------------------------------------

class MessageModel(Base):
    """Direct messages between learners and instructors."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    recipient_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    sender = relationship("UserModel", foreign_keys=[sender_id], back_populates="sent_messages")
    recipient = relationship("UserModel", foreign_keys=[recipient_id], back_populates="received_messages")


class GroupMessageModel(Base):
    """Messages in course group chats — all enrolled learners + instructor."""
    __tablename__ = "group_messages"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    course = relationship("CourseModel", back_populates="group_messages")
    sender = relationship("UserModel")


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


class ImageUrlResponse(BaseModel):
    url: str
    expires_in: int


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


class GoogleAuthRequest(BaseModel):
    credential: str


# Messaging Pydantic Models
class MessageCreate(BaseModel):
    recipient_id: int
    content: str


class MessageResponse(BaseModel):
    id: int
    sender_id: int
    sender_name: str
    sender_is_instructor: bool
    recipient_id: int
    recipient_name: str
    content: str
    is_read: bool
    created_at: datetime

    class Config:
        orm_mode = True


class GroupMessageCreate(BaseModel):
    content: str


class GroupMessageResponse(BaseModel):
    id: int
    course_id: int
    sender_id: int
    sender_name: str
    sender_is_instructor: bool
    content: str
    created_at: datetime

    class Config:
        orm_mode = True


class ConversationSummary(BaseModel):
    other_user_id: int
    other_user_name: str
    other_user_is_instructor: bool
    other_user_avatar: Optional[str] = None
    last_message: str
    last_message_at: datetime
    unread_count: int


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


SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440
VIDEO_TOKEN_EXPIRE_MINUTES = 60
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="ScienceTech Academy API")

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
    try:
        b2_client.delete_object(Bucket=B2_BUCKET_NAME, Key=filename)
        logger.info(f"File deleted from B2: {filename}")
    except Exception as e:
        logger.error(f"Error deleting file from B2: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


async def generate_presigned_url(filename: str, expiration: int = 3600) -> Optional[str]:
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
# Email Helper Functions
# ---------------------------------------------------------------------------

def _build_enrollment_email_html(user_name: str, course_title: str, course_url: str) -> str:
    """Build a styled HTML email body for enrollment confirmation."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Enrollment Confirmation</title>
</head>
<body style="margin:0;padding:0;background-color:#f4f6f9;font-family:Arial,Helvetica,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f6f9;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0"
               style="background-color:#ffffff;border-radius:8px;overflow:hidden;
                      box-shadow:0 2px 8px rgba(0,0,0,0.08);max-width:600px;width:100%;">

          <!-- Header -->
          <tr>
            <td style="background-color:#0B1A2E;padding:32px 40px;text-align:center;">
              <h1 style="margin:0;color:#C9A84C;font-size:22px;letter-spacing:1px;">
                {PLATFORM_NAME}
              </h1>
              <p style="margin:6px 0 0;color:#A8C8E8;font-size:13px;">{PLATFORM_TAGLINE}</p>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:40px 40px 24px;">
              <h2 style="margin:0 0 16px;color:#0B1A2E;font-size:20px;">
                🎉 You're enrolled!
              </h2>
              <p style="margin:0 0 12px;color:#444;font-size:15px;line-height:1.6;">
                Hi <strong>{user_name}</strong>,
              </p>
              <p style="margin:0 0 20px;color:#444;font-size:15px;line-height:1.6;">
                Congratulations! You have been successfully enrolled in:
              </p>

              <!-- Course box -->
              <table width="100%" cellpadding="0" cellspacing="0"
                     style="background-color:#f0f4ff;border-left:4px solid #C9A84C;
                            border-radius:4px;margin-bottom:24px;">
                <tr>
                  <td style="padding:16px 20px;">
                    <p style="margin:0;font-size:16px;font-weight:bold;color:#0B1A2E;">
                      {course_title}
                    </p>
                  </td>
                </tr>
              </table>

              <p style="margin:0 0 28px;color:#444;font-size:15px;line-height:1.6;">
                You can now access your course material, watch lessons, take quizzes,
                and track your progress — all at your own pace.
              </p>

              <!-- CTA Button -->
              <table cellpadding="0" cellspacing="0" style="margin-bottom:28px;">
                <tr>
                  <td style="background-color:#C9A84C;border-radius:6px;">
                    <a href="{course_url}"
                       style="display:inline-block;padding:14px 32px;color:#0B1A2E;
                              font-weight:bold;font-size:15px;text-decoration:none;
                              letter-spacing:0.5px;">
                      Start Learning →
                    </a>
                  </td>
                </tr>
              </table>

              <p style="margin:0;color:#888;font-size:13px;line-height:1.6;">
                If the button above doesn't work, copy and paste this link into your browser:<br/>
                <a href="{course_url}" style="color:#3a7bd5;">{course_url}</a>
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background-color:#f9f9f9;padding:20px 40px;border-top:1px solid #eee;
                       text-align:center;">
              <p style="margin:0;color:#aaa;font-size:12px;">
                © {datetime.utcnow().year} {PLATFORM_NAME} · {PLATFORM_DOMAIN}
              </p>
              <p style="margin:6px 0 0;color:#aaa;font-size:12px;">
                You received this email because you enrolled in a course on our platform.
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""


def _build_enrollment_email_text(user_name: str, course_title: str, course_url: str) -> str:
    """Plain-text fallback for enrollment confirmation email."""
    return (
        f"Hi {user_name},\n\n"
        f"Congratulations! You have been successfully enrolled in:\n\n"
        f"  {course_title}\n\n"
        f"Start learning now: {course_url}\n\n"
        f"Good luck on your learning journey!\n\n"
        f"— The {PLATFORM_NAME} Team\n"
        f"  {PLATFORM_DOMAIN}"
    )


def send_enrollment_email(user_email: str, user_name: str, course_title: str, course_id: int):
    """
    Send an enrollment confirmation email via SMTP.
    This is a synchronous function — call it inside a BackgroundTask.
    Does nothing (logs a warning) if email credentials are not configured.
    """
    if not EMAIL_ENABLED:
        logger.warning(
            "Email not configured (EMAIL_HOST_USER / EMAIL_HOST_PASSWORD missing). "
            f"Skipping enrollment email for {user_email}."
        )
        return

    course_url = f"https://{PLATFORM_DOMAIN}/courses/{course_id}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"You're enrolled in {course_title} — {PLATFORM_NAME}"
    msg["From"] = f"{EMAIL_FROM_NAME} <{EMAIL_HOST_USER}>"
    msg["To"] = user_email

    text_part = MIMEText(
        _build_enrollment_email_text(user_name, course_title, course_url), "plain", "utf-8"
    )
    html_part = MIMEText(
        _build_enrollment_email_html(user_name, course_title, course_url), "html", "utf-8"
    )
    # Clients render the last part first (prefer HTML)
    msg.attach(text_part)
    msg.attach(html_part)

    try:
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT, timeout=15) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
            smtp.sendmail(EMAIL_HOST_USER, user_email, msg.as_string())
        logger.info(f"Enrollment email sent to {user_email} for course '{course_title}'")
    except smtplib.SMTPAuthenticationError:
        logger.error(
            "SMTP authentication failed. Check EMAIL_HOST_USER and EMAIL_HOST_PASSWORD. "
            "For Gmail, use an App Password, not your account password."
        )
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error sending enrollment email to {user_email}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error sending enrollment email to {user_email}: {e}")


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

    try:
        resp = requests.post(auth_url, json=payload, headers=headers, timeout=30)
        logger.info(f"PesaPal auth response status: {resp.status_code}")
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

    try:
        expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00")).replace(tzinfo=None)
        expiry -= timedelta(minutes=5)
        logger.info(f"PesaPal token expires at {expiry}")
    except Exception as e:
        logger.warning(f"Could not parse expiry date '{expiry_str}': {e}")
        expiry = now + timedelta(minutes=50)

    _pesapal_token_cache = {"token": token, "expiry": expiry}
    logger.info("PesaPal token obtained and cached.")
    return token


def pesapal_register_ipn(token: str) -> str:
    global _pesapal_ipn_id

    if _pesapal_ipn_id:
        logger.info(f"Using cached IPN ID: {_pesapal_ipn_id}")
        return _pesapal_ipn_id

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

    try:
        resp = requests.post(ipn_url, json=payload, headers=headers, timeout=30)
        logger.info(f"IPN registration response status: {resp.status_code}")
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

    try:
        resp = requests.post(order_url, json=payload, headers=headers, timeout=30)
        logger.info(f"SubmitOrder response status: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"SubmitOrder response body (FULL): {json.dumps(data, indent=2)}")
    except requests.RequestException as e:
        logger.error(f"PesaPal SubmitOrderRequest error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response body: {e.response.text}")
        raise HTTPException(status_code=502, detail="Could not submit payment order to PesaPal.")

    error_field = data.get("error")
    if error_field is not None:
        if isinstance(error_field, dict):
            error_message = error_field.get("message") or error_field.get("errorMessage") or str(error_field)
        else:
            error_message = str(error_field)
        logger.error(f"PesaPal order error field: {error_field}")
        raise HTTPException(status_code=502, detail=f"PesaPal error: {error_message}")

    status_val = data.get("status")
    if status_val and str(status_val) not in ("200", "0"):
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
    status_url = f"{PESAPAL_BASE_URL}/api/Transactions/GetTransactionStatus"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    params = {"orderTrackingId": order_tracking_id}

    logger.info(f"Querying transaction status for {order_tracking_id}")

    try:
        resp = requests.get(status_url, headers=headers, params=params, timeout=30)
        logger.info(f"Transaction status response status: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Transaction status response body (FULL): {json.dumps(data, indent=2)}")
        return data
    except requests.RequestException as e:
        logger.error(f"PesaPal GetTransactionStatus error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response body: {e.response.text}")
        raise HTTPException(status_code=502, detail="Could not query payment status from PesaPal.")


def _enroll_user_in_course(
    db: Session,
    user_id: int,
    course_id: int,
    background_tasks: Optional[BackgroundTasks] = None,
):
    """
    Enroll a user in a course.  If the enrollment is new and background_tasks
    is provided, a confirmation email is queued as a background task.
    """
    existing = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == user_id,
        EnrollmentModel.course_id == course_id,
    ).first()
    if not existing:
        enrollment = EnrollmentModel(user_id=user_id, course_id=course_id)
        db.add(enrollment)
        db.commit()
        logger.info(f"User {user_id} enrolled in course {course_id} after payment.")

        # Queue enrollment email
        if background_tasks is not None:
            user = db.query(UserModel).filter(UserModel.id == user_id).first()
            course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
            if user and course:
                background_tasks.add_task(
                    send_enrollment_email,
                    user.email,
                    user.full_name,
                    course.title,
                    course_id,
                )
    else:
        logger.info(f"User {user_id} already enrolled in course {course_id}")


# ---------------------------------------------------------------------------
# Quiz Generation
# ---------------------------------------------------------------------------

async def generate_quiz(lesson_content: str) -> List[QuizQuestion]:
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


def _fetch_qr_png(verify_url: str, size: int = 120) -> Optional[bytes]:
    try:
        encoded = quote(verify_url, safe='')
        chart_url = (
            f"https://chart.googleapis.com/chart"
            f"?cht=qr&chs={size}x{size}&chl={encoded}&choe=UTF-8&chld=M|2"
        )
        resp = requests.get(chart_url, timeout=8)
        if resp.status_code == 200 and resp.content:
            logger.info(f"QR code fetched successfully ({len(resp.content)} bytes)")
            return resp.content
    except Exception as e:
        logger.warning(f"QR code fetch failed: {e}")
    return None


class _QRPlaceholder(Flowable):
    def __init__(self, size_pt: float, url: str):
        super().__init__()
        self.size_pt = size_pt
        self.url = url
        self.width = size_pt
        self.height = size_pt

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.HexColor('#C9A84C'))
        c.setLineWidth(1.5)
        c.rect(0, 0, self.size_pt, self.size_pt, fill=1, stroke=1)
        c.setFillColor(colors.HexColor('#0B1A2E'))
        c.setFont("Helvetica-Bold", 6)
        c.drawCentredString(self.size_pt / 2, self.size_pt * 0.55, "SCAN TO VERIFY")
        c.setFont("Helvetica", 5)
        c.drawCentredString(self.size_pt / 2, self.size_pt * 0.42, "QR code unavailable")
        c.setFont("Helvetica", 4)
        short = self.url.split("/")[-1][:16] if "/" in self.url else self.url[:16]
        c.drawCentredString(self.size_pt / 2, self.size_pt * 0.28, short.upper())
        c.restoreState()


class _HRule(Flowable):
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
    canvas.saveState()
    W, H = A4

    canvas.setFillColor(colors.HexColor('#0B1A2E'))
    canvas.rect(0, 0, W, H, fill=1, stroke=0)

    canvas.setFillColor(colors.HexColor('#C9A84C'))
    canvas.rect(0, H - 58, W, 58, fill=1, stroke=0)

    canvas.setFillColor(colors.HexColor('#C9A84C'))
    canvas.rect(0, 0, W, 58, fill=1, stroke=0)

    canvas.setStrokeColor(colors.white)
    canvas.setLineWidth(1)
    canvas.line(0, H - 62, W, H - 62)
    canvas.line(0, 62, W, 62)

    canvas.setFillColor(colors.HexColor('#C9A84C'))
    canvas.rect(0, 58, 14, H - 116, fill=1, stroke=0)
    canvas.rect(W - 14, 58, 14, H - 116, fill=1, stroke=0)

    canvas.setStrokeColor(colors.HexColor('#FFFFFF'))
    canvas.setLineWidth(0.5)
    canvas.line(18, 62, 18, H - 62)
    canvas.line(W - 18, 62, W - 18, H - 62)

    gold = colors.HexColor('#C9A84C')
    corners = [(14, H - 58), (W - 14, H - 58), (14, 58), (W - 14, 58)]
    for cx, cy in corners:
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
        canvas.setFillColor(gold)
        canvas.circle(cx, cy, 3, fill=1, stroke=0)

    canvas.setStrokeColor(colors.HexColor('#1A2E4A'))
    canvas.setLineWidth(1)
    cx, cy = W / 2, H / 2
    for r in (100, 108, 116):
        canvas.circle(cx, cy, r, fill=0, stroke=1)

    canvas.setStrokeColor(colors.HexColor('#1A2E4A'))
    canvas.setLineWidth(0.5)
    for i in range(24):
        angle = math.radians(i * 15)
        x1 = cx + 112 * math.cos(angle)
        y1 = cy + 112 * math.sin(angle)
        x2 = cx + 120 * math.cos(angle)
        y2 = cy + 120 * math.sin(angle)
        canvas.line(x1, y1, x2, y2)

    canvas.setFillColor(colors.HexColor('#0B1A2E'))
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawCentredString(W / 2, H - 38, "S C I E N C E T E C H   A C A D E M Y")

    canvas.setFillColor(colors.HexColor('#0B1A2E'))
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(W / 2, 22, f"{PLATFORM_DOMAIN}  ·  {PLATFORM_TAGLINE}")

    canvas.restoreState()


def create_certificate_pdf(user: UserModel, course: CourseModel, certificate_hash: str) -> BytesIO:
    buffer = BytesIO()

    left_margin = 38
    right_margin = 38
    top_margin = 72
    bottom_margin = 72

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
    )

    W, H = A4
    usable_width = W - left_margin - right_margin

    GOLD       = colors.HexColor('#C9A84C')
    LIGHT_GOLD = colors.HexColor('#E8D5A3')
    WHITE      = colors.white
    LIGHT_BLUE = colors.HexColor('#A8C8E8')
    SILVER     = colors.HexColor('#C0C0C0')

    def _style(name, font='Helvetica', size=12, color=WHITE, space_before=0,
               space_after=8, align=TA_CENTER, leading=None):
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

    style_eyebrow   = _style('eyebrow',   'Helvetica',      9,  LIGHT_GOLD,   0,  4)
    style_big_title = _style('bigtitle',  'Helvetica-Bold', 28, GOLD,         0, 10, leading=32)
    style_present   = _style('present',   'Helvetica',      11, SILVER,       6,  4)
    style_name      = _style('name',      'Helvetica-Bold', 26, WHITE,        4, 10, leading=30)
    style_for_comp  = _style('forcomp',   'Helvetica',      10, SILVER,       0,  4)
    style_course    = _style('course',    'Helvetica-Bold', 17, GOLD,         4, 10, leading=21)
    style_body      = _style('body',      'Helvetica',      10, SILVER,       4,  4)
    style_date      = _style('date',      'Helvetica',      10, LIGHT_GOLD,   2,  2)
    style_hash      = _style('hash',      'Helvetica',       7, colors.HexColor('#5A7A9A'), 8, 0)
    style_sig_name  = _style('signame',   'Helvetica-Bold', 10, WHITE,        2,  0)
    style_sig_role  = _style('sigrole',   'Helvetica',       8, SILVER,       0,  0)
    style_qr_label  = _style('qrlabel',   'Helvetica',       7, LIGHT_GOLD,   4,  0)

    elements = []

    try:
        logo_resp = requests.get(
            "https://raw.githubusercontent.com/Dariusmumbere/elearning/main/logo.png",
            timeout=5
        )
        if logo_resp.status_code == 200:
            logo_img = Image(BytesIO(logo_resp.content))
            aspect = logo_img.imageWidth / logo_img.imageHeight
            logo_img.drawWidth  = 0.65 * inch
            logo_img.drawHeight = 0.65 * inch / aspect
            logo_img.hAlign = 'CENTER'
            elements.append(logo_img)
            elements.append(Spacer(1, 6))
    except Exception as e:
        logger.warning(f"Could not load logo for certificate: {e}")

    elements.append(Paragraph("— OFFICIAL CERTIFICATE OF ACHIEVEMENT —", style_eyebrow))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph("ScienceTech Academy", style_big_title))
    elements.append(Spacer(1, 6))
    elements.append(_HRule(usable_width, thickness=2, color=GOLD))
    elements.append(Spacer(1, 2))
    elements.append(_HRule(usable_width, thickness=0.5, color=LIGHT_GOLD))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("This is to proudly certify that", style_present))
    elements.append(Spacer(1, 6))

    name_table = Table(
        [[Paragraph(user.full_name, style_name)]],
        colWidths=[usable_width],
    )
    name_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), colors.HexColor('#102040')),
        ('BOX',           (0, 0), (-1, -1), 1.5, GOLD),
        ('LEFTPADDING',   (0, 0), (-1, -1), 18),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 18),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(name_table)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("has successfully completed all requirements for the course", style_for_comp))
    elements.append(Spacer(1, 6))

    course_table = Table(
        [[Paragraph(course.title, style_course)]],
        colWidths=[usable_width],
    )
    course_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), colors.HexColor('#0D2240')),
        ('BOX',           (0, 0), (-1, -1), 0.75, LIGHT_GOLD),
        ('LINEBELOW',     (0, 0), (-1, -1), 3, GOLD),
        ('LEFTPADDING',   (0, 0), (-1, -1), 24),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 24),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(course_table)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "demonstrating dedication, commitment, and mastery of the curriculum.",
        style_body
    ))

    completion_date = datetime.utcnow().strftime("%B %d, %Y")
    elements.append(Paragraph(f"Issued on  {completion_date}", style_date))
    elements.append(Spacer(1, 12))
    elements.append(_HRule(usable_width, thickness=0.5, color=LIGHT_GOLD))
    elements.append(Spacer(1, 2))
    elements.append(_HRule(usable_width, thickness=2, color=GOLD))
    elements.append(Spacer(1, 14))

    verify_url = f"https://{PLATFORM_DOMAIN}/verify/{certificate_hash}"
    qr_png_bytes = _fetch_qr_png(verify_url, size=130)
    qr_size_pt = 1.2 * inch

    if qr_png_bytes:
        qr_img = Image(BytesIO(qr_png_bytes))
        qr_img.drawWidth  = qr_size_pt
        qr_img.drawHeight = qr_size_pt
        qr_img.hAlign = 'CENTER'
        qr_cell = [qr_img, Paragraph("Scan to verify", style_qr_label)]
    else:
        qr_cell = [_QRPlaceholder(qr_size_pt, verify_url), Paragraph("Scan to verify", style_qr_label)]

    def _sig_cell(name, role):
        return [
            Paragraph("______________________", _style('sigline', 'Helvetica', 10, SILVER, 0, 2)),
            Paragraph(name, style_sig_name),
            Paragraph(role, style_sig_role),
        ]

    sig_col_w = (usable_width - qr_size_pt - 12) / 3

    sig_data = [[
        _sig_cell("Course Instructor", "Lead Instructor"),
        _sig_cell("ScienceTech Academy", "Issuing Authority"),
        _sig_cell("Academic Director", "Platform Director"),
        qr_cell,
    ]]
    sig_table = Table(sig_data, colWidths=[sig_col_w, sig_col_w, sig_col_w, qr_size_pt + 12])
    sig_table.setStyle(TableStyle([
        ('ALIGN',          (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',         (0, 0), (2, 0),   'TOP'),
        ('VALIGN',         (3, 0), (3, 0),   'MIDDLE'),
        ('LEFTPADDING',    (0, 0), (-1, -1), 4),
        ('RIGHTPADDING',   (0, 0), (-1, -1), 4),
    ]))
    elements.append(sig_table)
    elements.append(Spacer(1, 12))

    id_table = Table(
        [[Paragraph(
            f"Certificate ID: {certificate_hash.upper()}  ·  Verify at {PLATFORM_DOMAIN}/verify/{certificate_hash}",
            style_hash
        )]],
        colWidths=[usable_width],
    )
    id_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), colors.HexColor('#061220')),
        ('BOX',           (0, 0), (-1, -1), 0.5, colors.HexColor('#1E3A5A')),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
    ]))
    elements.append(id_table)

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
# WebSocket Connection Manager
# ===========================================================================

class ConnectionManager:
    def __init__(self):
        self.direct_connections: Dict[int, Set[WebSocket]] = {}
        self.group_connections: Dict[int, Set[WebSocket]] = {}

    async def connect_direct(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        if user_id not in self.direct_connections:
            self.direct_connections[user_id] = set()
        self.direct_connections[user_id].add(websocket)
        logger.info(f"Direct WS connected: user_id={user_id}")

    def disconnect_direct(self, websocket: WebSocket, user_id: int):
        if user_id in self.direct_connections:
            self.direct_connections[user_id].discard(websocket)
            if not self.direct_connections[user_id]:
                del self.direct_connections[user_id]
        logger.info(f"Direct WS disconnected: user_id={user_id}")

    async def connect_group(self, websocket: WebSocket, course_id: int):
        await websocket.accept()
        if course_id not in self.group_connections:
            self.group_connections[course_id] = set()
        self.group_connections[course_id].add(websocket)
        logger.info(f"Group WS connected: course_id={course_id}")

    def disconnect_group(self, websocket: WebSocket, course_id: int):
        if course_id in self.group_connections:
            self.group_connections[course_id].discard(websocket)
            if not self.group_connections[course_id]:
                del self.group_connections[course_id]
        logger.info(f"Group WS disconnected: course_id={course_id}")

    async def send_to_user(self, user_id: int, message: dict):
        if user_id in self.direct_connections:
            dead = set()
            for ws in self.direct_connections[user_id]:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.add(ws)
            for ws in dead:
                self.direct_connections[user_id].discard(ws)

    async def broadcast_to_group(self, course_id: int, message: dict):
        if course_id in self.group_connections:
            dead = set()
            for ws in self.group_connections[course_id]:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.add(ws)
            for ws in dead:
                self.group_connections[course_id].discard(ws)


manager = ConnectionManager()


def _get_user_from_ws_token(token: str, db: Session) -> Optional[UserModel]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            return None
        return get_user_by_email(db, email)
    except JWTError:
        return None


# ===========================================================================
# Routes
# ===========================================================================

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


@app.post("/auth/google", response_model=Token)
async def google_auth(payload: GoogleAuthRequest, db: Session = Depends(get_db)):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google Sign-In is not configured.")

    try:
        id_info = id_token.verify_oauth2_token(
            payload.credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except ValueError as e:
        logger.error(f"Google token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid Google token.")

    email = id_info.get("email")
    full_name = id_info.get("name", email.split("@")[0])

    if not email:
        raise HTTPException(status_code=400, detail="Google account has no email address.")

    user = get_user_by_email(db, email)
    if not user:
        random_password = uuid.uuid4().hex
        user = UserModel(
            email=email,
            hashed_password=get_password_hash(random_password),
            full_name=full_name,
            is_instructor=False,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"New user created via Google Sign-In: {email}")
    else:
        logger.info(f"Existing user signed in via Google: {email}")

    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}


# ---------------------------------------------------------------------------
# Profile Routes
# ---------------------------------------------------------------------------

@app.get("/users/me/profile", response_model=ProfileResponse)
async def get_my_profile(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
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
        if user.profile_image_filename:
            try:
                await delete_from_b2(user.profile_image_filename)
            except Exception as e:
                logger.warning(f"Could not delete old profile image: {e}")
        user.profile_image_filename = await upload_to_b2(profile_image, "avatars")

    db.commit()
    db.refresh(user)

    profile_image_url = None
    if user.profile_image_filename:
        profile_image_url = f"/b2-proxy/{user.profile_image_filename}"

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

_VIDEO_CHUNK_SIZE = 512 * 1024


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

    try:
        head = b2_client.head_object(Bucket=B2_BUCKET_NAME, Key=object_key)
        file_size = head["ContentLength"]
        content_type = head.get("ContentType") or "video/mp4"
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
            logger.error(f"Stream error for '{object_key}' after {bytes_sent} bytes: {exc}")

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
# Quiz Endpoints
# ---------------------------------------------------------------------------

@app.post("/lessons/{lesson_id}/generate-quiz", response_model=QuizResponse)
async def generate_quiz_for_lesson(
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
        "platform": PLATFORM_NAME,
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
# PesaPal Payment Routes
# ---------------------------------------------------------------------------

@app.post("/payments/initiate/{course_id}", response_model=PaymentInitiateResponse)
async def initiate_payment(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    course = db.query(CourseModel).filter(
        CourseModel.id == course_id,
        CourseModel.is_published == True,
    ).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found.")

    existing_enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == course_id,
    ).first()
    if existing_enrollment:
        raise HTTPException(status_code=400, detail="You are already enrolled in this course.")

    existing_completed_payment = db.query(PaymentModel).filter(
        PaymentModel.user_id == current_user.id,
        PaymentModel.course_id == course_id,
        PaymentModel.status == "COMPLETED",
    ).first()
    if existing_completed_payment:
        _enroll_user_in_course(db, current_user.id, course_id)
        raise HTTPException(
            status_code=400,
            detail="Payment already completed. You have been enrolled in the course."
        )

    merchant_reference = f"crs-{course_id}-usr-{current_user.id}-{uuid.uuid4().hex[:8]}"
    pesapal_token = pesapal_get_token()
    ipn_id = pesapal_register_ipn(pesapal_token)

    callback_url = f"{PESAPAL_CALLBACK_URL}?merchant_reference={merchant_reference}"
    name_parts = current_user.full_name.strip().split(" ", 1)
    first_name = name_parts[0]
    last_name = name_parts[1] if len(name_parts) > 1 else ""

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

    if not redirect_url:
        raise HTTPException(status_code=502, detail="PesaPal did not return a payment URL.")

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

    return PaymentInitiateResponse(
        redirect_url=redirect_url,
        merchant_reference=merchant_reference,
        order_tracking_id=order_tracking_id,
    )


@app.get("/payments/callback")
async def payment_callback(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    params = dict(request.query_params)
    order_tracking_id = params.get("OrderTrackingId") or params.get("orderTrackingId")
    merchant_reference = params.get("OrderMerchantReference") or params.get("merchant_reference")

    if not merchant_reference:
        redirect_url = f"{PESAPAL_CALLBACK_URL.split('/payment')[0]}/payment/result?status=error&message=Missing+payment+reference"
        return RedirectResponse(url=redirect_url, status_code=302)

    payment = db.query(PaymentModel).filter(PaymentModel.merchant_reference == merchant_reference).first()
    if not payment:
        redirect_url = f"{PESAPAL_CALLBACK_URL.split('/payment')[0]}/payment/result?status=error&message=Payment+not+found"
        return RedirectResponse(url=redirect_url, status_code=302)

    if order_tracking_id and not payment.order_tracking_id:
        payment.order_tracking_id = order_tracking_id
        db.commit()

    tracking_id = payment.order_tracking_id or order_tracking_id
    if not tracking_id:
        redirect_url = f"{PESAPAL_CALLBACK_URL.split('/payment')[0]}/payment/result?status=pending&course_id={payment.course_id}"
        return RedirectResponse(url=redirect_url, status_code=302)

    try:
        pesapal_token = pesapal_get_token()
        status_data = pesapal_get_transaction_status(pesapal_token, tracking_id)
        payment_status_code = status_data.get("payment_status_description", "").upper()
        pesapal_status = status_data.get("status_code")

        if payment_status_code == "COMPLETED" or pesapal_status == 1:
            payment.status = "COMPLETED"
            db.commit()
            _enroll_user_in_course(db, payment.user_id, payment.course_id, background_tasks)
            redirect_url = f"{PESAPAL_CALLBACK_URL.split('/payment')[0]}/payment/result?status=success&course_id={payment.course_id}"
            return RedirectResponse(url=redirect_url, status_code=302)
        elif payment_status_code in ("FAILED", "INVALID") or pesapal_status in (0, 2):
            payment.status = "FAILED"
            db.commit()
            redirect_url = f"{PESAPAL_CALLBACK_URL.split('/payment')[0]}/payment/result?status=failed&course_id={payment.course_id}"
            return RedirectResponse(url=redirect_url, status_code=302)
        else:
            redirect_url = f"{PESAPAL_CALLBACK_URL.split('/payment')[0]}/payment/result?status=pending&course_id={payment.course_id}"
            return RedirectResponse(url=redirect_url, status_code=302)
    except Exception as e:
        logger.error(f"Error verifying payment on callback: {e}", exc_info=True)
        redirect_url = f"{PESAPAL_CALLBACK_URL.split('/payment')[0]}/payment/result?status=pending&course_id={payment.course_id}"
        return RedirectResponse(url=redirect_url, status_code=302)


@app.get("/payments/ipn")
async def payment_ipn(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    params = dict(request.query_params)
    order_tracking_id = params.get("orderTrackingId") or params.get("OrderTrackingId")
    merchant_reference = params.get("orderMerchantReference") or params.get("OrderMerchantReference")
    order_notification_type = params.get("orderNotificationType")

    if not order_tracking_id:
        return {"status": "ignored", "reason": "missing orderTrackingId"}

    payment = None
    if merchant_reference:
        payment = db.query(PaymentModel).filter(PaymentModel.merchant_reference == merchant_reference).first()
    if not payment and order_tracking_id:
        payment = db.query(PaymentModel).filter(PaymentModel.order_tracking_id == order_tracking_id).first()
    if not payment:
        return {"status": "ignored", "reason": "payment not found"}

    if order_tracking_id and not payment.order_tracking_id:
        payment.order_tracking_id = order_tracking_id
        db.commit()

    try:
        pesapal_token = pesapal_get_token()
        status_data = pesapal_get_transaction_status(pesapal_token, order_tracking_id)
        payment_status_code = status_data.get("payment_status_description", "").upper()
        pesapal_status = status_data.get("status_code")

        if payment_status_code == "COMPLETED" or pesapal_status == 1:
            if payment.status != "COMPLETED":
                payment.status = "COMPLETED"
                payment.updated_at = datetime.utcnow()
                db.commit()
                _enroll_user_in_course(db, payment.user_id, payment.course_id, background_tasks)
        elif payment_status_code in ("FAILED", "INVALID") or pesapal_status in (0, 2):
            if payment.status != "FAILED":
                payment.status = "FAILED"
                payment.updated_at = datetime.utcnow()
                db.commit()
    except Exception as e:
        logger.error(f"IPN processing error: {e}", exc_info=True)

    return {
        "orderNotificationType": order_notification_type,
        "orderTrackingId": order_tracking_id,
        "orderMerchantReference": merchant_reference,
    }


@app.get("/payments/verify/{merchant_reference}", response_model=PaymentStatusResponse)
async def verify_payment(
    merchant_reference: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    payment = db.query(PaymentModel).filter(
        PaymentModel.merchant_reference == merchant_reference,
        PaymentModel.user_id == current_user.id,
    ).first()
    if not payment:
        raise HTTPException(status_code=404, detail="Payment record not found.")

    if payment.status == "PENDING" and payment.order_tracking_id:
        try:
            pesapal_token = pesapal_get_token()
            status_data = pesapal_get_transaction_status(pesapal_token, payment.order_tracking_id)
            payment_status_code = status_data.get("payment_status_description", "").upper()
            pesapal_status = status_data.get("status_code")

            if payment_status_code == "COMPLETED" or pesapal_status == 1:
                payment.status = "COMPLETED"
                payment.updated_at = datetime.utcnow()
                db.commit()
                _enroll_user_in_course(db, payment.user_id, payment.course_id, background_tasks)
            elif payment_status_code in ("FAILED", "INVALID") or pesapal_status in (0, 2):
                payment.status = "FAILED"
                payment.updated_at = datetime.utcnow()
                db.commit()
        except Exception as e:
            logger.error(f"Error re-querying PesaPal in verify endpoint: {e}", exc_info=True)

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
    payments = db.query(PaymentModel).filter(
        PaymentModel.user_id == current_user.id
    ).order_by(PaymentModel.created_at.desc()).all()

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
    if not current_user.is_instructor:
        raise HTTPException(status_code=403, detail="Only instructors can create courses")

    image_filename = None
    if image_file:
        allowed_image_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if image_file.content_type not in allowed_image_types:
            raise HTTPException(status_code=400, detail="Invalid file type.")
        image_filename = await upload_to_b2(image_file, "courses")

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
    courses = db.query(CourseModel).filter(CourseModel.is_published == True).offset(skip).limit(limit).all()

    course_list = []
    for course in courses:
        image_url = await get_course_image_url(course)
        instructor = db.query(UserModel).filter(UserModel.id == course.instructor_id).first()
        instructor_name = instructor.full_name if instructor else "Unknown Instructor"
        course_list.append(_build_course_response(course, instructor_name, image_url))

    return course_list


@app.get("/courses/{course_id}", response_model=Course)
async def read_course(course_id: int, db: Session = Depends(get_db)):
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
            raise HTTPException(status_code=400, detail="Invalid file type.")
        if course.image_filename:
            await delete_from_b2(course.image_filename)
        course.image_filename = await upload_to_b2(image_file, "courses")

    db.commit()
    db.refresh(course)

    image_url = await get_course_image_url(course)
    instructor = db.query(UserModel).filter(UserModel.id == course.instructor_id).first()
    instructor_name = instructor.full_name if instructor else "Unknown Instructor"
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
            raise HTTPException(status_code=400, detail="Invalid file type.")
        video_filename = await upload_to_b2(video_file, "lessons")

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
            raise HTTPException(status_code=400, detail="Invalid file type.")
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
# Enrollment Routes
# ---------------------------------------------------------------------------

@app.post("/enroll/{course_id}")
def enroll_in_course(
    course_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
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

    # Send enrollment confirmation email in the background
    background_tasks.add_task(
        send_enrollment_email,
        current_user.email,
        current_user.full_name,
        course.title,
        course_id,
    )

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
            Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")} | {PLATFORM_NAME}
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


# ---------------------------------------------------------------------------
# AI Consultation
# ---------------------------------------------------------------------------

@app.post("/lessons/{lesson_id}/ai-consult", response_model=AIConsultResponse)
async def ai_consult(
    lesson_id: int,
    request: AIConsultRequest,
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

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question is too long (max 1000 characters)")

    lesson_context = lesson.content or "No written content available for this lesson."

    system_prompt = f"""You are an expert AI tutor for {PLATFORM_NAME}.
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

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
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
        raise HTTPException(status_code=500, detail="Failed to get AI response. Please try again.")


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
# B2 Proxy
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
            raise HTTPException(status_code=404, detail="File not found")
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


# ===========================================================================
# Messaging REST Endpoints
# ===========================================================================

@app.post("/messages/direct", response_model=MessageResponse)
async def send_direct_message(
    payload: MessageCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    recipient = db.query(UserModel).filter(UserModel.id == payload.recipient_id).first()
    if not recipient:
        raise HTTPException(status_code=404, detail="Recipient not found")

    if payload.recipient_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot send a message to yourself")

    if not current_user.is_instructor and not recipient.is_instructor:
        raise HTTPException(status_code=403, detail="Learners can only message instructors")

    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(content) > 2000:
        raise HTTPException(status_code=400, detail="Message too long (max 2000 characters)")

    msg = MessageModel(
        sender_id=current_user.id,
        recipient_id=payload.recipient_id,
        content=content,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)

    msg_data = {
        "type": "direct_message",
        "id": msg.id,
        "sender_id": current_user.id,
        "sender_name": current_user.full_name,
        "sender_is_instructor": current_user.is_instructor,
        "recipient_id": payload.recipient_id,
        "recipient_name": recipient.full_name,
        "content": content,
        "is_read": False,
        "created_at": msg.created_at.isoformat(),
    }

    await manager.send_to_user(current_user.id, msg_data)
    await manager.send_to_user(payload.recipient_id, msg_data)

    return MessageResponse(
        id=msg.id,
        sender_id=msg.sender_id,
        sender_name=current_user.full_name,
        sender_is_instructor=current_user.is_instructor,
        recipient_id=msg.recipient_id,
        recipient_name=recipient.full_name,
        content=msg.content,
        is_read=msg.is_read,
        created_at=msg.created_at,
    )


@app.get("/messages/direct/{other_user_id}", response_model=List[MessageResponse])
async def get_direct_conversation(
    other_user_id: int,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    other_user = db.query(UserModel).filter(UserModel.id == other_user_id).first()
    if not other_user:
        raise HTTPException(status_code=404, detail="User not found")

    messages = db.query(MessageModel).filter(
        ((MessageModel.sender_id == current_user.id) & (MessageModel.recipient_id == other_user_id)) |
        ((MessageModel.sender_id == other_user_id) & (MessageModel.recipient_id == current_user.id))
    ).order_by(MessageModel.created_at.asc()).all()

    db.query(MessageModel).filter(
        MessageModel.sender_id == other_user_id,
        MessageModel.recipient_id == current_user.id,
        MessageModel.is_read == False,
    ).update({"is_read": True})
    db.commit()

    return [
        MessageResponse(
            id=m.id,
            sender_id=m.sender_id,
            sender_name=m.sender.full_name,
            sender_is_instructor=m.sender.is_instructor,
            recipient_id=m.recipient_id,
            recipient_name=m.recipient.full_name,
            content=m.content,
            is_read=m.is_read,
            created_at=m.created_at,
        )
        for m in messages
    ]


@app.get("/messages/conversations", response_model=List[ConversationSummary])
async def get_conversations(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    all_messages = db.query(MessageModel).filter(
        (MessageModel.sender_id == current_user.id) | (MessageModel.recipient_id == current_user.id)
    ).order_by(MessageModel.created_at.desc()).all()

    seen: Dict[int, ConversationSummary] = {}
    for m in all_messages:
        other_id = m.recipient_id if m.sender_id == current_user.id else m.sender_id
        if other_id in seen:
            continue
        other = db.query(UserModel).filter(UserModel.id == other_id).first()
        if not other:
            continue
        unread = db.query(MessageModel).filter(
            MessageModel.sender_id == other_id,
            MessageModel.recipient_id == current_user.id,
            MessageModel.is_read == False,
        ).count()
        avatar_url = f"/b2-proxy/{other.profile_image_filename}" if other.profile_image_filename else None
        seen[other_id] = ConversationSummary(
            other_user_id=other_id,
            other_user_name=other.full_name,
            other_user_is_instructor=other.is_instructor,
            other_user_avatar=avatar_url,
            last_message=m.content[:80],
            last_message_at=m.created_at,
            unread_count=unread,
        )
    return list(seen.values())


@app.get("/messages/unread-count")
async def get_unread_count(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    count = db.query(MessageModel).filter(
        MessageModel.recipient_id == current_user.id,
        MessageModel.is_read == False,
    ).count()
    return {"unread_count": count}


# ---------------------------------------------------------------------------
# Group (Course) Messaging REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/messages/group/{course_id}", response_model=List[GroupMessageResponse])
async def get_group_messages(
    course_id: int,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(100, le=200),
):
    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    is_instructor = course.instructor_id == current_user.id
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == course_id,
    ).first()
    if not is_instructor and not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    messages = db.query(GroupMessageModel).filter(
        GroupMessageModel.course_id == course_id
    ).order_by(GroupMessageModel.created_at.asc()).limit(limit).all()

    return [
        GroupMessageResponse(
            id=m.id,
            course_id=m.course_id,
            sender_id=m.sender_id,
            sender_name=m.sender.full_name,
            sender_is_instructor=m.sender.is_instructor,
            content=m.content,
            created_at=m.created_at,
        )
        for m in messages
    ]


@app.post("/messages/group/{course_id}", response_model=GroupMessageResponse)
async def send_group_message(
    course_id: int,
    payload: GroupMessageCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    is_instructor = course.instructor_id == current_user.id
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == course_id,
    ).first()
    if not is_instructor and not enrollment:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")

    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(content) > 2000:
        raise HTTPException(status_code=400, detail="Message too long (max 2000 characters)")

    msg = GroupMessageModel(
        course_id=course_id,
        sender_id=current_user.id,
        content=content,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)

    msg_data = {
        "type": "group_message",
        "id": msg.id,
        "course_id": course_id,
        "sender_id": current_user.id,
        "sender_name": current_user.full_name,
        "sender_is_instructor": current_user.is_instructor,
        "content": content,
        "created_at": msg.created_at.isoformat(),
    }

    await manager.broadcast_to_group(course_id, msg_data)

    return GroupMessageResponse(
        id=msg.id,
        course_id=msg.course_id,
        sender_id=msg.sender_id,
        sender_name=current_user.full_name,
        sender_is_instructor=current_user.is_instructor,
        content=msg.content,
        created_at=msg.created_at,
    )


@app.get("/messages/my-instructors")
async def get_my_instructors(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    enrollments = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id
    ).all()
    instructor_ids = set()
    for e in enrollments:
        course = db.query(CourseModel).filter(CourseModel.id == e.course_id).first()
        if course:
            instructor_ids.add(course.instructor_id)

    instructors = []
    for iid in instructor_ids:
        user = db.query(UserModel).filter(UserModel.id == iid).first()
        if user:
            avatar_url = f"/b2-proxy/{user.profile_image_filename}" if user.profile_image_filename else None
            instructors.append({
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "avatar_url": avatar_url,
            })
    return instructors


@app.get("/messages/my-learners")
async def get_my_learners(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.is_instructor:
        raise HTTPException(status_code=403, detail="Only instructors can access this endpoint")

    courses = db.query(CourseModel).filter(CourseModel.instructor_id == current_user.id).all()
    learner_ids = set()
    for course in courses:
        for enrollment in course.enrollments:
            learner_ids.add(enrollment.user_id)

    learners = []
    for lid in learner_ids:
        user = db.query(UserModel).filter(UserModel.id == lid).first()
        if user:
            avatar_url = f"/b2-proxy/{user.profile_image_filename}" if user.profile_image_filename else None
            learners.append({
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "avatar_url": avatar_url,
            })
    return learners


# ===========================================================================
# WebSocket Endpoints
# ===========================================================================

@app.websocket("/ws/direct/{user_id}")
async def websocket_direct(
    websocket: WebSocket,
    user_id: int,
    token: str = Query(...),
    db: Session = Depends(get_db),
):
    user = _get_user_from_ws_token(token, db)
    if not user or user.id != user_id:
        await websocket.close(code=4001)
        return

    await manager.connect_direct(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            recipient_id = data.get("recipient_id")
            content = (data.get("content") or "").strip()
            if not recipient_id or not content:
                continue

            recipient = db.query(UserModel).filter(UserModel.id == recipient_id).first()
            if not recipient:
                continue
            if not user.is_instructor and not recipient.is_instructor:
                continue
            if len(content) > 2000:
                content = content[:2000]

            msg = MessageModel(
                sender_id=user.id,
                recipient_id=recipient_id,
                content=content,
            )
            db.add(msg)
            db.commit()
            db.refresh(msg)

            msg_data = {
                "type": "direct_message",
                "id": msg.id,
                "sender_id": user.id,
                "sender_name": user.full_name,
                "sender_is_instructor": user.is_instructor,
                "recipient_id": recipient_id,
                "recipient_name": recipient.full_name,
                "content": content,
                "is_read": False,
                "created_at": msg.created_at.isoformat(),
            }
            await manager.send_to_user(user.id, msg_data)
            await manager.send_to_user(recipient_id, msg_data)

    except WebSocketDisconnect:
        manager.disconnect_direct(websocket, user_id)


@app.websocket("/ws/group/{course_id}")
async def websocket_group(
    websocket: WebSocket,
    course_id: int,
    token: str = Query(...),
    db: Session = Depends(get_db),
):
    user = _get_user_from_ws_token(token, db)
    if not user:
        await websocket.close(code=4001)
        return

    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    if not course:
        await websocket.close(code=4004)
        return

    is_instructor = course.instructor_id == user.id
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == user.id,
        EnrollmentModel.course_id == course_id,
    ).first()
    if not is_instructor and not enrollment:
        await websocket.close(code=4003)
        return

    await manager.connect_group(websocket, course_id)
    try:
        while True:
            data = await websocket.receive_json()
            content = (data.get("content") or "").strip()
            if not content:
                continue
            if len(content) > 2000:
                content = content[:2000]

            msg = GroupMessageModel(
                course_id=course_id,
                sender_id=user.id,
                content=content,
            )
            db.add(msg)
            db.commit()
            db.refresh(msg)

            msg_data = {
                "type": "group_message",
                "id": msg.id,
                "course_id": course_id,
                "sender_id": user.id,
                "sender_name": user.full_name,
                "sender_is_instructor": user.is_instructor,
                "content": content,
                "created_at": msg.created_at.isoformat(),
            }
            await manager.broadcast_to_group(course_id, msg_data)

    except WebSocketDisconnect:
        manager.disconnect_group(websocket, course_id)


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
        return {"status": "error", "detail": e.detail}


@app.get("/debug/email")
async def debug_email():
    """Check email configuration status (no credentials exposed)."""
    return {
        "email_enabled": EMAIL_ENABLED,
        "email_host": EMAIL_HOST,
        "email_port": EMAIL_PORT,
        "email_from_name": EMAIL_FROM_NAME,
        "email_host_user_configured": bool(EMAIL_HOST_USER),
        "email_password_configured": bool(EMAIL_HOST_PASSWORD),
    }


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "healthy", "platform": PLATFORM_NAME, "timestamp": datetime.utcnow()}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {PLATFORM_NAME} API...")
    db = SessionLocal()
    try:
        Base.metadata.create_all(bind=engine)
        migrate_has_quiz_default(db)
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS bio TEXT;"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS profile_image_filename VARCHAR(255);"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS location VARCHAR(255);"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS website VARCHAR(255);"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS phone VARCHAR(50);"))
        logger.info("Startup migrations completed successfully")
        if EMAIL_ENABLED:
            logger.info(f"Email notifications enabled (SMTP: {EMAIL_HOST}:{EMAIL_PORT}, from: {EMAIL_HOST_USER})")
        else:
            logger.warning(
                "Email notifications DISABLED — set EMAIL_HOST_USER and EMAIL_HOST_PASSWORD env vars to enable."
            )
        try:
            token = pesapal_get_token()
            ipn_id = pesapal_register_ipn(token)
            logger.info(f"PesaPal IPN pre-registered on startup. ipn_id={ipn_id}")
        except Exception as pe:
            logger.warning(f"PesaPal IPN pre-registration failed on startup: {pe}")
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
