from fastapi import FastAPI, Depends, HTTPException, status, Form, UploadFile, File, BackgroundTasks, Request, Query, Header, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON, func
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
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, Response
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
import base64
from io import BytesIO
import hashlib
import asyncio
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
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://elearning_ltdb_user:FD2SwB0zrdhQjFEOPLCHXr9hytKwsaKD@dpg-d7jqsgt7vvec73dq64q0-a/elearning_ltdb")

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

    courses = relationship("CourseModel", back_populates="instructor")
    enrollments = relationship("EnrollmentModel", back_populates="user")
    quiz_attempts = relationship("QuizAttemptModel", back_populates="user")
    certificates = relationship("CertificateModel", back_populates="user")


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


# ── UPDATED: image_filename now included so the frontend can build proxy URLs ──
class Course(CourseBase):
    id: int
    instructor_id: int
    created_at: datetime
    instructor_name: Optional[str] = None
    is_published: bool
    image_url: Optional[str] = None
    image_filename: Optional[str] = None   # ← ADDED

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


async def generate_presigned_url(filename: str, expiration: int = 3600):
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


async def get_course_image_url(course: CourseModel) -> Optional[str]:
    """Return the /b2-proxy/ URL for a course image."""
    if not course.image_filename:
        return None
    return f"/b2-proxy/{course.image_filename}"


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


def create_certificate_pdf(user: UserModel, course: CourseModel, certificate_hash: str) -> BytesIO:
    """Create a visually appealing certificate PDF."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CertificateTitle', parent=styles['Title'],
        fontSize=36, spaceAfter=30, alignment=TA_CENTER,
        textColor=colors.HexColor('#2C3E50'), fontName='Helvetica-Bold',
    )
    subtitle_style = ParagraphStyle(
        'CertificateSubtitle', parent=styles['Heading2'],
        fontSize=20, spaceAfter=20, alignment=TA_CENTER,
        textColor=colors.HexColor('#34495E'), fontName='Helvetica',
    )
    normal_style = ParagraphStyle(
        'CertificateNormal', parent=styles['Normal'],
        fontSize=16, spaceAfter=12, alignment=TA_CENTER,
        textColor=colors.HexColor('#7F8C8D'), fontName='Helvetica',
    )
    name_style = ParagraphStyle(
        'CertificateName', parent=styles['Heading1'],
        fontSize=42, spaceAfter=30, alignment=TA_CENTER,
        textColor=colors.HexColor('#2980B9'), fontName='Helvetica-Bold',
    )
    small_style = ParagraphStyle(
        'CertificateSmall', parent=styles['Normal'],
        fontSize=10, alignment=TA_CENTER,
        textColor=colors.HexColor('#BDC3C7'), fontName='Helvetica',
    )

    def draw_border(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#E67E22'))
        canvas.setLineWidth(2)
        x, y = doc.leftMargin / 2, doc.bottomMargin / 2
        width = doc.pagesize[0] - doc.leftMargin
        height = doc.pagesize[1] - doc.bottomMargin
        canvas.rect(x, y, width, height, stroke=1, fill=0)
        canvas.restoreState()

    try:
        logo_url = "https://raw.githubusercontent.com/Dariusmumbere/elearning/main/logo.png"
        response = requests.get(logo_url)
        if response.status_code == 200:
            logo_img = Image(BytesIO(response.content))
            logo_img.drawHeight = 1.5 * inch * logo_img.drawHeight / logo_img.drawWidth
            logo_img.drawWidth = 1.5 * inch
            logo_img.hAlign = 'CENTER'
            elements.append(logo_img)
            elements.append(Spacer(1, 0.2 * inch))
    except Exception as e:
        logger.warning(f"Could not load logo: {e}")
        elements.append(Paragraph("eLearning Platform", subtitle_style))

    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("CERTIFICATE OF ACHIEVEMENT", title_style))
    elements.append(Spacer(1, 0.2 * inch))

    line = Table([['']], colWidths=[4 * inch], rowHeights=[0.02 * inch])
    line.setStyle(TableStyle([('LINEABOVE', (0, 0), (-1, -1), 2, colors.HexColor('#3498DB'))]))
    elements.append(line)
    elements.append(Spacer(1, 0.4 * inch))

    elements.append(Paragraph("This is to certify that", normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    name_bg = Table([[Paragraph(user.full_name.upper(), name_style)]], colWidths=[6 * inch])
    name_bg.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F5F5F5')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(name_bg)
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("has successfully completed the course", normal_style))
    elements.append(Spacer(1, 0.3 * inch))

    course_bg = Table([[Paragraph(f'"{course.title}"', subtitle_style)]], colWidths=[6 * inch])
    course_bg.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F8FF')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#B0C4DE')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(course_bg)
    elements.append(Spacer(1, 0.4 * inch))

    completion_date = datetime.utcnow().strftime("%B %d, %Y")
    elements.append(Paragraph(f"Awarded on {completion_date}", normal_style))
    elements.append(Spacer(1, 0.5 * inch))

    signature_data = [
        ["_________________________", "_________________________"],
        ["Course Instructor", "Platform Director"],
    ]
    signature_table = Table(signature_data, colWidths=[3 * inch, 3 * inch])
    signature_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#7F8C8D')),
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#2C3E50')),
    ]))
    elements.append(signature_table)
    elements.append(Spacer(1, 0.4 * inch))

    cert_id_bg = Table([[Paragraph(f"Certificate ID: {certificate_hash}", small_style)]], colWidths=[6 * inch])
    cert_id_bg.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F5F5F5')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#E0E0E0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(cert_id_bg)

    doc.build(elements, onFirstPage=draw_border, onLaterPages=draw_border)
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


def ensure_course_image_in_b2(db: Session, course_id: int):
    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    if not course:
        return None
    if course.image_url and not course.image_filename:
        logger.info(f"Course {course_id} has image_url but no image_filename. Migration needed.")
    return course.image_filename


# ---------------------------------------------------------------------------
# Helper: build a Course response dict consistently
# ---------------------------------------------------------------------------

def _build_course_response(course: CourseModel, instructor_name: str, image_url: Optional[str]) -> dict:
    """
    Single place that constructs the dict fed to the Course Pydantic model.
    image_filename is always included so the frontend can build /b2-proxy/ URLs.
    """
    return {
        "id": course.id,
        "title": course.title,
        "description": course.description,
        "instructor_id": course.instructor_id,
        "instructor_name": instructor_name,
        "created_at": course.created_at,
        "is_published": course.is_published,
        "image_url": image_url,
        "image_filename": course.image_filename,   # ← always exposed
    }


# ===========================================================================
# Routes
# ===========================================================================

# ---------------------------------------------------------------------------
# Video Streaming
# ---------------------------------------------------------------------------

@app.get("/stream/video/{filename:path}")
async def stream_video(
    filename: str,
    request: Request,
    token: str = Query(..., description="JWT token for video access"),
    db: Session = Depends(get_db),
):
    logger.info(f"Video streaming request for filename: {filename}")
    logger.info(f"Token received: {token[:20]}...")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        lesson_id = payload.get("lesson_id")

        logger.info(f"Token payload: {payload}")
        logger.info(f"Requested filename: {filename}")

        if not user_id or not lesson_id:
            raise HTTPException(status_code=401, detail="Invalid video token")

        user = db.query(UserModel).filter(UserModel.id == user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
        if not lesson:
            raise HTTPException(status_code=404, detail="Lesson not found")

        logger.info(f"Lesson filename from DB: {lesson.video_filename}")

        enrollment = db.query(EnrollmentModel).filter(
            EnrollmentModel.user_id == user_id,
            EnrollmentModel.course_id == lesson.module.course_id,
        ).first()
        if not enrollment:
            raise HTTPException(status_code=403, detail="Not enrolled in this course")

        if not lesson.video_filename:
            raise HTTPException(status_code=404, detail="Video not found for this lesson")

        object_key = lesson.video_filename
        logger.info(f"Using object key from database: {object_key}")

        try:
            head_response = b2_client.head_object(Bucket=B2_BUCKET_NAME, Key=object_key)
            file_size = head_response['ContentLength']
            content_type = head_response['ContentType'] or 'video/mp4'
            logger.info(f"File exists in B2. Size: {file_size}, Type: {content_type}")
        except ClientError as e:
            logger.error(f"B2 HEAD Error: {e}")
            raise HTTPException(status_code=404, detail=f"Video file not found in storage: {str(e)}")

        range_header = request.headers.get("Range")
        logger.info(f"Range header: {range_header}")

        if range_header:
            try:
                range_type, range_value = range_header.split('=')
                if range_type.strip().lower() != 'bytes':
                    raise HTTPException(status_code=416, detail="Invalid Range Type")

                start_str, end_str = range_value.split('-')
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1

                if start >= file_size or end >= file_size or start > end:
                    raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")

                content_length = end - start + 1
                byte_range = f"bytes={start}-{end}"
                logger.info(f"Requesting byte range: {byte_range}, content length: {content_length}")

                response = b2_client.get_object(
                    Bucket=B2_BUCKET_NAME, Key=object_key, Range=byte_range
                )

                headers = {
                    'Content-Range': f'bytes {start}-{end}/{file_size}',
                    'Accept-Ranges': 'bytes',
                    'Content-Length': str(content_length),
                    'Content-Type': content_type,
                    'Cache-Control': 'public, max-age=3600',
                }

                def generate_chunks():
                    try:
                        chunk_size = 1024 * 1024
                        with response['Body'] as stream:
                            while True:
                                chunk = stream.read(chunk_size)
                                if not chunk:
                                    break
                                yield chunk
                    except Exception as e:
                        logger.error(f"Error during chunk streaming: {e}")
                        raise

                return StreamingResponse(
                    generate_chunks(), status_code=206, headers=headers, media_type=content_type
                )

            except ClientError as e:
                logger.error(f"Range Streaming Error: {e}")
                raise HTTPException(status_code=500, detail="Error streaming video (range)")
            except ValueError as e:
                logger.error(f"Invalid range format: {e}")
                raise HTTPException(status_code=416, detail="Invalid Range Format")

        # Full video stream
        logger.info("No range header, streaming full video")
        response = b2_client.get_object(Bucket=B2_BUCKET_NAME, Key=object_key)

        headers = {
            'Accept-Ranges': 'bytes',
            'Content-Length': str(file_size),
            'Content-Type': content_type,
            'Cache-Control': 'public, max-age=3600',
        }

        def generate_full_chunks():
            try:
                chunk_size = 1024 * 1024
                with response['Body'] as stream:
                    while True:
                        chunk = stream.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
            except Exception as e:
                logger.error(f"Error during full video streaming: {e}")
                raise

        return StreamingResponse(generate_full_chunks(), headers=headers, media_type=content_type)

    except JWTError as e:
        logger.error(f"JWT Decode Error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired video token")
    except Exception as e:
        logger.error(f"Unexpected error in stream_video: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


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
    else:
        image_filename = "courses/default-course-image.jpg"

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
# Enrollment Routes
# ---------------------------------------------------------------------------

@app.post("/enroll/{course_id}")
def enroll_in_course(
    course_id: int,
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
async def b2_proxy(filename: str):
    """Proxy private B2 files — used for course images and other assets."""
    try:
        presigned_url = await generate_presigned_url(filename)
        if not presigned_url:
            raise HTTPException(status_code=404, detail="File not found")

        response = requests.get(presigned_url, stream=True)
        if response.status_code == 200:
            return StreamingResponse(
                response.iter_content(chunk_size=8192),
                media_type=response.headers.get('content-type', 'application/octet-stream'),
                headers={
                    'Content-Disposition': response.headers.get('content-disposition', 'inline'),
                    'Content-Length': response.headers.get('content-length', ''),
                    'Cache-Control': 'public, max-age=3600',
                },
            )
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Error accessing B2 file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error accessing file: {str(e)}")


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

    b2_info = {}
    if course.image_filename:
        try:
            head_response = b2_client.head_object(Bucket=B2_BUCKET_NAME, Key=course.image_filename)
            b2_info = {
                "exists": True,
                "size": head_response['ContentLength'],
                "content_type": head_response['ContentType'],
                "last_modified": head_response['LastModified'],
                "image_filename": course.image_filename,
            }
        except ClientError as e:
            b2_info = {"exists": False, "error": str(e), "image_filename": course.image_filename}

    return {
        "course_id": course_id,
        "course_title": course.title,
        "has_image_filename": bool(course.image_filename),
        "image_url": course.image_url,
        "image_filename": course.image_filename,
        "b2_info": b2_info,
        "b2_proxy_url": f"/b2-proxy/{course.image_filename}" if course.image_filename else None,
    }


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
        logger.info("Startup migrations completed successfully")
    except Exception as e:
        logger.error(f"Startup migration error: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    Base.metadata.create_all(bind=engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)
