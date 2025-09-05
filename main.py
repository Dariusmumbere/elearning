# main.py (Updated with AI Quiz System)
from fastapi import FastAPI, Depends, HTTPException, status, Form, UploadFile, File, BackgroundTasks, Request, Query, Header, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional, List, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
import shutil
import uuid
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import boto3
from botocore.exceptions import ClientError
import io
import requests
from urllib.parse import quote
import logging
import google.generativeai as genai
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBIZbO5KawONRW9-JCBoIQ7vX5EhSKFhNM")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Backblaze B2 Configuration - PRIVATE BUCKET
B2_BUCKET_NAME = "uploads-dir"
B2_ENDPOINT_URL = "https://s3.us-east-005.backblazeb2.com"
B2_KEY_ID = "0055ca7845641d30000000002"
B2_APPLICATION_KEY = "K005NNeGM9r28ujQ3jvNEQy2zUiu0TI"

# Initialize B2 client
b2_client = boto3.client(
    's3',
    endpoint_url=B2_ENDPOINT_URL,
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APPLICATION_KEY
)

# Database configuration (using your credentials)
DATABASE_URL = "postgresql://blog_0bcu_user:RXAJHCfB4v6iU9gaNBHrA06QmCzZxLFK@dpg-d2nbbmq4d50c73e5ovug-a/blog_0bcu"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models (updated with quiz tables)
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
    has_quiz = Column(Boolean, default=False)
    
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

# New Quiz Models
class QuizAttemptModel(Base):
    __tablename__ = "quiz_attempts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    lesson_id = Column(Integer, ForeignKey("lessons.id"))
    questions = Column(JSON)  # Store the quiz questions and correct answers
    user_answers = Column(JSON)  # Store user's answers
    score = Column(Integer)  # Number of correct answers
    passed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("UserModel", back_populates="quiz_attempts")
    lesson = relationship("LessonModel", back_populates="quiz_attempts")

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models (updated with quiz models)
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

class Course(CourseBase):
    id: int
    instructor_id: int
    created_at: datetime
    is_published: bool
    image_url: Optional[str] = None
    
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
    has_quiz: Optional[bool] = False

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
    
    class Config:
        orm_mode = True

class VideoTokenResponse(BaseModel):
    token: str
    expires_at: datetime

# Quiz Models
class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: int  # Index of the correct option (0-based)

class QuizRequest(BaseModel):
    lesson_content: str

class QuizResponse(BaseModel):
    questions: List[QuizQuestion]

class QuizSubmission(BaseModel):
    answers: List[int]  # List of selected option indices

class QuizResult(BaseModel):
    score: int
    total: int
    passed: bool
    correct_answers: List[int]

class QuizAttemptResponse(BaseModel):
    id: int
    user_id: int
    lesson_id: int
    score: int
    passed: bool
    created_at: datetime
    
    class Config:
        orm_mode = True
        
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

# Auth setup
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
VIDEO_TOKEN_EXPIRE_MINUTES = 60  # Shorter lifespan for video tokens

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="eLearning Platform API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
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
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_video_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=VIDEO_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

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

# B2 Helper Functions
async def upload_to_b2(file: UploadFile, folder: str) -> str:
    """Upload a file to Backblaze B2 and return the URL"""
    try:
        # Generate unique filename
        file_extension = file.filename.split(".")[-1]
        filename = f"{folder}/{uuid.uuid4()}.{file_extension}"
        
        # Read file content
        file_content = await file.read()
        
        # Upload to B2
        b2_client.put_object(
            Bucket=B2_BUCKET_NAME,
            Key=filename,
            Body=file_content,
            ContentType=file.content_type
        )
        
        return filename
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

async def delete_from_b2(filename: str):
    """Delete a file from Backblaze B2"""
    try:
        b2_client.delete_object(Bucket=B2_BUCKET_NAME, Key=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

async def generate_presigned_url(filename: str, expiration: int = 3600):
    """Generate a presigned URL for private B2 objects"""
    try:
        url = b2_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': B2_BUCKET_NAME,
                'Key': filename
            },
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Error generating presigned URL: {str(e)}")

# Quiz Helper Functions
async def generate_quiz(lesson_content: str) -> List[QuizQuestion]:
    """Generate a quiz using Gemini AI based on lesson content"""
    try:
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
        
        Lesson Content:
        {lesson_content}
        """
        
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_str = response.text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            quiz_data = json.loads(json_str)
            
            # Validate the structure
            if "questions" in quiz_data and len(quiz_data["questions"]) == 5:
                return quiz_data["questions"]
            else:
                raise ValueError("Invalid quiz format from AI")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse AI response: {e}")
            logger.error(f"AI response: {response.text}")
            
            # Fallback: create a simple quiz
            return [
                {
                    "question": "What was the main topic of this lesson?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": 0
                }
                for _ in range(5)
            ]
            
    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        # Fallback quiz
        return [
            {
                "question": f"Question {i+1} about the lesson content",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": 0
            }
            for i in range(5)
        ]

# NEW: Robust Video Streaming Endpoint with Debugging Logs
@app.get("/stream/video/{filename:path}")
async def stream_video(
    filename: str,
    request: Request,
    token: str = Query(..., description="JWT token for video access"),
    db: Session = Depends(get_db)
):
    logger.info(f"Video streaming request received for filename: {filename}")
    logger.info(f"Token received: {token[:20]}...")  # Log first 20 chars of token
    
    try:
        # Decode token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        lesson_id = payload.get("lesson_id")

        # Debug logs
        logger.info(f"Token payload: {payload}")
        logger.info(f"Requested filename: {filename}")

        if not user_id or not lesson_id:
            logger.error("Invalid video token: missing user_id or lesson_id")
            raise HTTPException(status_code=401, detail="Invalid video token")

        # Fetch user and lesson
        user = db.query(UserModel).filter(UserModel.id == user_id).first()
        if not user:
            logger.error(f"User not found with ID: {user_id}")
            raise HTTPException(status_code=401, detail="User not found")

        lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
        if not lesson:
            logger.error(f"Lesson not found with ID: {lesson_id}")
            raise HTTPException(status_code=404, detail="Lesson not found")

        # Debug log
        logger.info(f"Lesson filename from DB: {lesson.video_filename}")

        # Check enrollment
        enrollment = db.query(EnrollmentModel).filter(
            EnrollmentModel.user_id == user_id,
            EnrollmentModel.course_id == lesson.module.course_id
        ).first()

        if not enrollment:
            logger.error(f"User {user_id} not enrolled in course {lesson.module.course_id}")
            raise HTTPException(status_code=403, detail="Not enrolled in this course")

        # Compare filenames (robust check using pathlib)
        requested_filename = Path(filename).name
        expected_filename = Path(lesson.video_filename).name

        if requested_filename != expected_filename:
            logger.error(f"Filename mismatch: requested {requested_filename}, expected {expected_filename}")
            raise HTTPException(status_code=403, detail="Invalid video access")

        # Confirm file exists in B2 before streaming
        try:
            head_response = b2_client.head_object(Bucket=B2_BUCKET_NAME, Key=lesson.video_filename)
            file_size = head_response['ContentLength']
            content_type = head_response['ContentType']
            logger.info(f"File exists in B2! Size: {file_size} bytes, Type: {content_type}")
        except ClientError as e:
            logger.error(f"B2 HEAD Error: {e}")
            raise HTTPException(status_code=404, detail="Video file not found in B2")

        object_key = lesson.video_filename
        range_header = request.headers.get("Range")
        logger.info(f"Range header: {range_header}")

        # Handle range requests
        if range_header:
            try:
                # Parse range header
                range_type, range_value = range_header.split('=')
                if range_type.strip().lower() != 'bytes':
                    logger.error(f"Invalid range type: {range_type}")
                    raise HTTPException(status_code=416, detail="Invalid Range Type")

                start_str, end_str = range_value.split('-')
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1

                # Validate range
                if start >= file_size or end >= file_size or start > end:
                    logger.error(f"Invalid range: {start}-{end} for file size {file_size}")
                    raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")

                # Calculate content length
                content_length = end - start + 1
                byte_range = f"bytes={start}-{end}"

                logger.info(f"Requesting byte range: {byte_range}, content length: {content_length}")

                # Get the range from B2
                response = b2_client.get_object(
                    Bucket=B2_BUCKET_NAME,
                    Key=object_key,
                    Range=byte_range
                )

                headers = {
                    'Content-Range': f'bytes {start}-{end}/{file_size}',
                    'Accept-Ranges': 'bytes',
                    'Content-Length': str(content_length),
                    'Content-Type': content_type
                }

                logger.info(f"Streaming range response with headers: {headers}")

                # Stream the response in chunks
                def generate_chunks():
                    try:
                        chunk_size = 1024 * 1024  # 1MB chunks
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
                    generate_chunks(),
                    status_code=206,
                    headers=headers,
                    media_type=content_type
                )

            except ClientError as e:
                logger.error(f"Range Streaming Error: {e}")
                raise HTTPException(status_code=500, detail="Error streaming video (range)")
            except ValueError as e:
                logger.error(f"Invalid range format: {e}")
                raise HTTPException(status_code=416, detail="Invalid Range Format")

        # Full video stream (if no range header)
        try:
            logger.info("No range header, streaming full video")
            response = b2_client.get_object(Bucket=B2_BUCKET_NAME, Key=object_key)

            headers = {
                'Accept-Ranges': 'bytes',
                'Content-Length': str(file_size),
                'Content-Type': content_type
            }

            logger.info(f"Streaming full video with headers: {headers}")

            # Stream the response in chunks
            def generate_full_chunks():
                try:
                    chunk_size = 1024 * 1024  # 1MB chunks
                    with response['Body'] as stream:
                        while True:
                            chunk = stream.read(chunk_size)
                            if not chunk:
                                break
                            yield chunk
                except Exception as e:
                    logger.error(f"Error during full video streaming: {e}")
                    raise

            return StreamingResponse(
                generate_full_chunks(),
                headers=headers,
                media_type=content_type
            )

        except ClientError as e:
            logger.error(f"Full Streaming Error: {e}")
            raise HTTPException(status_code=500, detail="Error streaming video")

    except JWTError as e:
        logger.error(f"JWT Decode Error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired video token")
    except Exception as e:
        logger.error(f"Unexpected error in stream_video: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
        

# NEW: Get video access token endpoint
@app.get("/video-token/{lesson_id}", response_model=VideoTokenResponse)
async def get_video_token(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Video token request for lesson {lesson_id} from user {current_user.id}")
    
    # Verify user has access to this lesson
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        logger.error(f"Lesson not found: {lesson_id}")
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    # Check if user is enrolled in the course
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id
    ).first()
    
    if not enrollment:
        logger.error(f"User {current_user.id} not enrolled in course {lesson.module.course_id}")
        raise HTTPException(status_code=403, detail="Not enrolled in this course")
    
    # Create a video token
    expires_delta = timedelta(minutes=VIDEO_TOKEN_EXPIRE_MINUTES)
    token_data = {
        "sub": "video_access",
        "user_id": current_user.id,
        "lesson_id": lesson_id
    }
    access_token = create_video_token(token_data, expires_delta=expires_delta)
    
    logger.info(f"Video token generated for user {current_user.id}, lesson {lesson_id}")
    return {
        "token": access_token,
        "expires_at": datetime.utcnow() + expires_delta
    }

# NEW: Quiz Endpoints
@app.post("/lessons/{lesson_id}/generate-quiz", response_model=QuizResponse)
async def generate_quiz_for_lesson(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Generating quiz for lesson {lesson_id} by user {current_user.id}")
    
    # Verify user has access to this lesson
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        logger.error(f"Lesson not found: {lesson_id}")
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    # Check if user is enrolled in the course
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id
    ).first()
    
    if not enrollment:
        logger.error(f"User {current_user.id} not enrolled in course {lesson.module.course_id}")
        raise HTTPException(status_code=403, detail="Not enrolled in this course")
    
    # Generate quiz using AI
    quiz_questions = await generate_quiz(lesson.content)
    
    return {"questions": quiz_questions}

@app.post("/lessons/{lesson_id}/submit-quiz", response_model=QuizResult)
async def submit_quiz(
    lesson_id: int,
    submission: QuizSubmission,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Submitting quiz for lesson {lesson_id} by user {current_user.id}")
    
    # Verify user has access to this lesson
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        logger.error(f"Lesson not found: {lesson_id}")
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    # Check if user is enrolled in the course
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id
    ).first()
    
    if not enrollment:
        logger.error(f"User {current_user.id} not enrolled in course {lesson.module.course_id}")
        raise HTTPException(status_code=403, detail="Not enrolled in this course")
    
    # Generate the same quiz again to get correct answers
    quiz_questions = await generate_quiz(lesson.content)
    
    # Calculate score
    score = 0
    correct_answers = []
    
    for i, question in enumerate(quiz_questions):
        correct_answers.append(question["correct_answer"])
        if i < len(submission.answers) and submission.answers[i] == question["correct_answer"]:
            score += 1
    
    # Check if passed (at least 3 correct answers)
    passed = score >= 3
    
    # Save quiz attempt
    quiz_attempt = QuizAttemptModel(
        user_id=current_user.id,
        lesson_id=lesson_id,
        questions=quiz_questions,
        user_answers=submission.answers,
        score=score,
        passed=passed
    )
    
    db.add(quiz_attempt)
    db.commit()
    db.refresh(quiz_attempt)
    
    logger.info(f"Quiz submitted: score {score}/5, passed: {passed}")
    
    return {
        "score": score,
        "total": 5,
        "passed": passed,
        "correct_answers": correct_answers
    }

@app.get("/lessons/{lesson_id}/quiz-attempts", response_model=List[QuizAttemptResponse])
async def get_quiz_attempts(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Getting quiz attempts for lesson {lesson_id} by user {current_user.id}")
    
    # Verify user has access to this lesson
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        logger.error(f"Lesson not found: {lesson_id}")
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    # Check if user is enrolled in the course
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id
    ).first()
    
    if not enrollment:
        logger.error(f"User {current_user.id} not enrolled in course {lesson.module.course_id}")
        raise HTTPException(status_code=403, detail="Not enrolled in this course")
    
    # Get user's quiz attempts for this lesson
    attempts = db.query(QuizAttemptModel).filter(
        QuizAttemptModel.user_id == current_user.id,
        QuizAttemptModel.lesson_id == lesson_id
    ).order_by(QuizAttemptModel.created_at.desc()).all()
    
    return attempts

# All the existing routes from your previous implementation...
# (Auth routes, course routes, module routes, lesson routes, etc.)

# Auth routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    logger.info(f"Login attempt for user: {form_data.username}")
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        logger.error(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    logger.info(f"Successful login for user: {form_data.username}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=User)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    logger.info(f"Creating new user: {user.email}")
    db_user = get_user_by_email(db, email=user.email)
    if db_user:
        logger.error(f"User already exists: {user.email}")
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    db_user = UserModel(
        email=user.email, 
        hashed_password=hashed_password, 
        full_name=user.full_name,
        is_instructor=user.is_instructor
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info(f"User created successfully: {user.email}")
    return db_user

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    logger.info(f"User data requested for: {current_user.email}")
    return current_user

# Course routes
@app.post("/courses/", response_model=Course)
async def create_course(
    title: str = Form(...),
    description: str = Form(None),
    image_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Creating course: {title} by user: {current_user.email}")
    if not current_user.is_instructor:
        logger.error(f"Non-instructor attempt to create course: {current_user.email}")
        raise HTTPException(status_code=403, detail="Only instructors can create courses")
    
    # Handle image upload if provided
    image_filename = None
    if image_file:
        # Validate file type
        allowed_image_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if image_file.content_type not in allowed_image_types:
            logger.error(f"Invalid image type: {image_file.content_type}")
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only JPEG, PNG, GIF, and WebP images are allowed."
            )
        
        # Upload to Backblaze B2
        image_filename = await upload_to_b2(image_file, "courses")
        logger.info(f"Course image uploaded: {image_filename}")
    
    db_course = CourseModel(
        title=title,
        description=description,
        instructor_id=current_user.id,
        image_filename=image_filename
    )
    db.add(db_course)
    db.commit()
    db.refresh(db_course)
    
    # Generate presigned URL for the image
    image_url = None
    if image_filename:
        image_url = await generate_presigned_url(image_filename)
    
    # Return course with image URL
    course_data = {
        "id": db_course.id,
        "title": db_course.title,
        "description": db_course.description,
        "instructor_id": db_course.instructor_id,
        "created_at": db_course.created_at,
        "is_published": db_course.is_published,
        "image_url": image_url
    }
    
    logger.info(f"Course created successfully: {db_course.id}")
    return course_data

@app.get("/courses/", response_model=List[Course])
async def read_courses(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    logger.info(f"Fetching courses, skip: {skip}, limit: {limit}")
    courses = db.query(CourseModel).filter(CourseModel.is_published == True).offset(skip).limit(limit).all()
    
    # Convert ORM objects to dicts and generate presigned URLs for images
    course_list = []
    for course in courses:
        image_url = None
        if course.image_filename:
            image_url = await generate_presigned_url(course.image_filename)
        
        course_dict = {
            "id": course.id,
            "title": course.title,
            "description": course.description,
            "instructor_id": course.instructor_id,
            "created_at": course.created_at,
            "is_published": course.is_published,
            "image_url": image_url
        }
        course_list.append(course_dict)
    
    logger.info(f"Returning {len(course_list)} courses")
    return course_list

@app.get("/courses/{course_id}", response_model=Course)
async def read_course(course_id: int, db: Session = Depends(get_db)):
    logger.info(f"Fetching course: {course_id}")
    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    if course is None:
        logger.error(f"Course not found: {course_id}")
        raise HTTPException(status_code=404, detail="Course not found")
    
    # Generate presigned URL for the image
    image_url = None
    if course.image_filename:
        image_url = await generate_presigned_url(course.image_filename)
    
    # Return the course with the correct image URL
    course_data = {
        "id": course.id,
        "title": course.title,
        "description": course.description,
        "instructor_id": course.instructor_id,
        "created_at": course.created_at,
        "is_published": course.is_published,
        "image_url": image_url
    }
    
    logger.info(f"Course found: {course_id}")
    return course_data

# Module routes
@app.post("/courses/{course_id}/modules/", response_model=Module)
def create_module(
    course_id: int,
    module: ModuleCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Creating module for course: {course_id} by user: {current_user.email}")
    if not current_user.is_instructor:
        logger.error(f"Non-instructor attempt to create module: {current_user.email}")
        raise HTTPException(status_code=403, detail="Only instructors can create modules")
    
    course = db.query(CourseModel).filter(CourseModel.id == course_id, CourseModel.instructor_id == current_user.id).first()
    if not course:
        logger.error(f"Course not found or permission denied: {course_id} for user: {current_user.id}")
        raise HTTPException(status_code=404, detail="Course not found or you don't have permission")
    
    db_module = ModuleModel(
        title=module.title,
        description=module.description,
        course_id=course_id,
        order=module.order
    )
    db.add(db_module)
    db.commit()
    db.refresh(db_module)
    logger.info(f"Module created successfully: {db_module.id}")
    return db_module

# Get modules for a course
@app.get("/courses/{course_id}/modules/", response_model=List[Module])
def get_course_modules(
    course_id: int,
    db: Session = Depends(get_db)
):
    logger.info(f"Fetching modules for course: {course_id}")
    modules = db.query(ModuleModel).filter(ModuleModel.course_id == course_id).order_by(ModuleModel.order).all()
    logger.info(f"Found {len(modules)} modules for course: {course_id}")
    return modules

# Lesson routes
@app.post("/modules/{module_id}/lessons/", response_model=LessonResponse)
async def create_lesson(
    module_id: int,
    title: str = Form(...),
    content: str = Form(None),
    order: int = Form(1),
    has_quiz: bool = Form(False),
    video_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Creating lesson for module: {module_id} by user: {current_user.email}")
    if not current_user.is_instructor:
        logger.error(f"Non-instructor attempt to create lesson: {current_user.email}")
        raise HTTPException(status_code=403, detail="Only instructors can create lessons")
    
    # Verify the module exists and belongs to the current instructor
    module = db.query(ModuleModel).join(CourseModel).filter(
        ModuleModel.id == module_id, 
        CourseModel.instructor_id == current_user.id
    ).first()
    
    if not module:
        logger.error(f"Module not found or permission denied: {module_id} for user: {current_user.id}")
        raise HTTPException(status_code=404, detail="Module not found or you don't have permission")
    
    # Handle video file upload if provided
    video_filename = None
    video_url = None
    
    if video_file:
        # Validate file type
        allowed_video_types = ["video/mp4", "video/mov", "video/avi", "video/webm", "video/quicktime"]
        if video_file.content_type not in allowed_video_types:
            logger.error(f"Invalid video type: {video_file.content_type}")
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only MP4, MOV, AVI, and WebM files are allowed."
            )
        
        # Upload to Backblaze B2
        video_filename = await upload_to_b2(video_file, "lessons")
        video_url = await generate_presigned_url(video_filename)
        logger.info(f"Video uploaded: {video_filename}")
    
    # Create the lesson
    db_lesson = LessonModel(
        title=title,
        content=content,
        module_id=module_id,
        order=order,
        has_quiz=has_quiz,
        video_url=video_url,
        video_filename=video_filename
    )
    
    db.add(db_lesson)
    db.commit()
    db.refresh(db_lesson)
    
    # Build the response
    response_data = {
        "id": db_lesson.id,
        "title": db_lesson.title,
        "content": db_lesson.content,
        "order": db_lesson.order,
        "module_id": db_lesson.module_id,
        "has_quiz": db_lesson.has_quiz,
        "video_url": db_lesson.video_url,
        "video_filename": db_lesson.video_filename
    }
    
    logger.info(f"Lesson created successfully: {db_lesson.id}")
    return response_data

# Get lessons for a module
@app.get("/modules/{module_id}/lessons/", response_model=List[LessonResponse])
async def get_module_lessons(
    module_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Fetching lessons for module: {module_id} by user: {current_user.email}")
    lessons = db.query(LessonModel).filter(LessonModel.module_id == module_id).order_by(LessonModel.order).all()
    
    # Check if user is enrolled in this course
    module = db.query(ModuleModel).filter(ModuleModel.id == module_id).first()
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == module.course_id
    ).first()
    
    # Build response with proper video URLs and progress
    response_lessons = []
    for lesson in lessons:
        completed = False
        if enrollment:
            progress = db.query(ProgressModel).filter(
                ProgressModel.enrollment_id == enrollment.id,
                ProgressModel.lesson_id == lesson.id,
                ProgressModel.completed == True
            ).first()
            completed = progress is not None
        
        # Generate presigned URL for video if it exists
        video_url = None
        if lesson.video_filename:
            video_url = await generate_presigned_url(lesson.video_filename)
        
        lesson_data = {
            "id": lesson.id,
            "title": lesson.title,
            "content": lesson.content,
            "order": lesson.order,
            "module_id": lesson.module_id,
            "has_quiz": lesson.has_quiz,
            "video_url": video_url,
            "video_filename": lesson.video_filename,
            "completed": completed
        }
            
        response_lessons.append(lesson_data)
    
    logger.info(f"Returning {len(response_lessons)} lessons for module: {module_id}")
    return response_lessons

# Publish course
@app.put("/courses/{course_id}/publish/")
def publish_course(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Publishing course: {course_id} by user: {current_user.email}")
    if not current_user.is_instructor:
        logger.error(f"Non-instructor attempt to publish course: {current_user.email}")
        raise HTTPException(status_code=403, detail="Only instructors can publish courses")
    
    course = db.query(CourseModel).filter(CourseModel.id == course_id, CourseModel.instructor_id == current_user.id).first()
    if not course:
        logger.error(f"Course not found or permission denied: {course_id} for user: {current_user.id}")
        raise HTTPException(status_code=404, detail="Course not found or you don't have permission")
    
    course.is_published = True
    db.commit()
    logger.info(f"Course published successfully: {course_id}")
    return {"message": "Course published successfully"}

# File upload to B2
@app.post("/upload/course-image/")
async def upload_course_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    logger.info(f"Uploading course image by user: {current_user.email}")
    if not current_user.is_instructor:
        logger.error(f"Non-instructor attempt to upload image: {current_user.email}")
        raise HTTPException(status_code=403, detail="Only instructors can upload images")
    
    # Upload to Backblaze B2
    filename = await upload_to_b2(file, "courses")
    
    # Generate presigned URL
    url = await generate_presigned_url(filename)
    
    logger.info(f"Image uploaded successfully: {filename}")
    return {"filename": filename, "url": url}

# Enrollment routes
@app.post("/enroll/{course_id}")
def enroll_in_course(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Enrolling user {current_user.email} in course: {course_id}")
    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    if not course:
        logger.error(f"Course not found: {course_id}")
        raise HTTPException(status_code=404, detail="Course not found")
    
    existing_enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == course_id
    ).first()
    
    if existing_enrollment:
        logger.error(f"User already enrolled: {current_user.email} in course: {course_id}")
        raise HTTPException(status_code=400, detail="Already enrolled in this course")
    
    enrollment = EnrollmentModel(user_id=current_user.id, course_id=course_id)
    db.add(enrollment)
    db.commit()
    
    logger.info(f"User enrolled successfully: {current_user.email} in course: {course_id}")
    return {"message": "Successfully enrolled in course"}

@app.get("/my-courses/", response_model=List[Course])
async def get_my_courses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Fetching courses for user: {current_user.email}")
    enrollments = db.query(EnrollmentModel).filter(EnrollmentModel.user_id == current_user.id).all()
    course_ids = [enrollment.course_id for enrollment in enrollments]
    courses = db.query(CourseModel).filter(CourseModel.id.in_(course_ids)).all()
    
    # Convert ORM objects to dicts and generate presigned URLs
    course_list = []
    for course in courses:
        image_url = None
        if course.image_filename:
            image_url = await generate_presigned_url(course.image_filename)
        
        course_dict = {
            "id": course.id,
            "title": course.title,
            "description": course.description,
            "instructor_id": course.instructor_id,
            "created_at": course.created_at,
            "is_published": course.is_published,
            "image_url": image_url
        }
        course_list.append(course_dict)
    
    logger.info(f"Returning {len(course_list)} courses for user: {current_user.email}")
    return course_list

# Get courses for the current instructor
@app.get("/instructor/courses/", response_model=List[Course])
async def get_instructor_courses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Fetching instructor courses for user: {current_user.email}")
    if not current_user.is_instructor:
        logger.error(f"Non-instructor attempt to access instructor courses: {current_user.email}")
        raise HTTPException(status_code=403, detail="Only instructors can access this endpoint")
    
    courses = db.query(CourseModel).filter(CourseModel.instructor_id == current_user.id).all()
    
    # Convert ORM objects to dicts and generate presigned URLs
    course_list = []
    for course in courses:
        image_url = None
        if course.image_filename:
            image_url = await generate_presigned_url(course.image_filename)
        
        course_dict = {
            "id": course.id,
            "title": course.title,
            "description": course.description,
            "instructor_id": course.instructor_id,
            "created_at": course.created_at,
            "is_published": course.is_published,
            "image_url": image_url
        }
        course_list.append(course_dict)
    
    logger.info(f"Returning {len(course_list)} instructor courses for user: {current_user.email}")
    return course_list

# Update course endpoint
@app.put("/courses/{course_id}", response_model=Course)
async def update_course(
    course_id: int,
    title: str = Form(None),
    description: str = Form(None),
    image_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Updating course: {course_id} by user: {current_user.email}")
    if not current_user.is_instructor:
        logger.error(f"Non-instructor attempt to update course: {current_user.email}")
        raise HTTPException(status_code=403, detail="Only instructors can update courses")
    
    course = db.query(CourseModel).filter(CourseModel.id == course_id, CourseModel.instructor_id == current_user.id).first()
    if not course:
        logger.error(f"Course not found or permission denied: {course_id} for user: {current_user.id}")
        raise HTTPException(status_code=404, detail="Course not found or you don't have permission")
    
    if title is not None:
        course.title = title
    if description is not None:
        course.description = description
    
    # Handle image upload if provided
    if image_file:
        # Validate file type
        allowed_image_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if image_file.content_type not in allowed_image_types:
            logger.error(f"Invalid image type: {image_file.content_type}")
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only JPEG, PNG, GIF, and WebP images are allowed."
            )
        
        # Delete old image from B2 if it exists
        if course.image_filename:
            await delete_from_b2(course.image_filename)
        
        # Upload new image to B2
        course.image_filename = await upload_to_b2(image_file, "courses")
    
    db.commit()
    db.refresh(course)
    
    # Generate presigned URL for the image
    image_url = None
    if course.image_filename:
        image_url = await generate_presigned_url(course.image_filename)
    
    # Return the course
    course_response = {
        "id": course.id,
        "title": course.title,
        "description": course.description,
        "instructor_id": course.instructor_id,
        "created_at": course.created_at,
        "is_published": course.is_published,
        "image_url": image_url
    }
    
    logger.info(f"Course updated successfully: {course_id}")
    return course_response

# Get a specific lesson
@app.get("/lessons/{lesson_id}", response_model=LessonResponse)
async def get_lesson(
    lesson_id: int,
    db: Session = Depends(get_db)
):
    logger.info(f"Fetching lesson: {lesson_id}")
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        logger.error(f"Lesson not found: {lesson_id}")
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    # Generate presigned URL for video if it exists
    video_url = None
    if lesson.video_filename:
        video_url = await generate_presigned_url(lesson.video_filename)
    
    response_data = {
        "id": lesson.id,
        "title": lesson.title,
        "content": lesson.content,
        "order": lesson.order,
        "module_id": lesson.module_id,
        "has_quiz": lesson.has_quiz,
        "video_url": video_url,
        "video_filename": lesson.video_filename
    }
    
    logger.info(f"Lesson found: {lesson_id}")
    return response_data

# Update a lesson
@app.put("/lessons/{lesson_id}", response_model=LessonResponse)
async def update_lesson(
    lesson_id: int,
    title: str = Form(None),
    content: str = Form(None),
    order: int = Form(None),
    has_quiz: bool = Form(None),
    video_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Updating lesson: {lesson_id} by user: {current_user.email}")
    if not current_user.is_instructor:
        logger.error(f"Non-instructor attempt to update lesson: {current_user.email}")
        raise HTTPException(status_code=403, detail="Only instructors can update lessons")
    
    lesson = db.query(LessonModel).join(ModuleModel).join(CourseModel).filter(
        LessonModel.id == lesson_id,
        CourseModel.instructor_id == current_user.id
    ).first()
    
    if not lesson:
        logger.error(f"Lesson not found or permission denied: {lesson_id} for user: {current_user.id}")
        raise HTTPException(status_code=404, detail="Lesson not found or you don't have permission")
    
    # Update fields if provided
    if title is not None:
        lesson.title = title
    if content is not None:
        lesson.content = content
    if order is not None:
        lesson.order = order
    if has_quiz is not None:
        lesson.has_quiz = has_quiz
    
    # Handle video file upload if provided
    if video_file:
        # Validate file type
        allowed_video_types = ["video/mp4", "video/mov", "video/avi", "video/webm", "video/quicktime"]
        if video_file.content_type not in allowed_video_types:
            logger.error(f"Invalid video type: {video_file.content_type}")
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only MP4, MOV, AVI, and WebM files are allowed."
            )
        
        # Remove old video file from B2 if it exists
        if lesson.video_filename:
            await delete_from_b2(lesson.video_filename)
        
        # Upload new video to B2
        lesson.video_filename = await upload_to_b2(video_file, "lessons")
        lesson.video_url = await generate_presigned_url(lesson.video_filename)
    
    db.commit()
    db.refresh(lesson)
    
    # Build the response
    response_data = {
        "id": lesson.id,
        "title": lesson.title,
        "content": lesson.content,
        "order": lesson.order,
        "module_id": lesson.module_id,
        "has_quiz": lesson.has_quiz,
        "video_url": lesson.video_url,
        "video_filename": lesson.video_filename
    }
    
    logger.info(f"Lesson updated successfully: {lesson_id}")
    return response_data

# Update a module
@app.put("/modules/{module_id}", response_model=Module)
def update_module(
    module_id: int,
    module_data: ModuleBase,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Updating module: {module_id} by user: {current_user.email}")
    if not current_user.is_instructor:
        logger.error(f"Non-instructor attempt to update module: {current_user.email}")
        raise HTTPException(status_code=403, detail="Only instructors can update modules")
    
    module = db.query(ModuleModel).join(CourseModel).filter(
        ModuleModel.id == module_id,
        CourseModel.instructor_id == current_user.id
    ).first()
    
    if not module:
        logger.error(f"Module not found or permission denied: {module_id} for user: {current_user.id}")
        raise HTTPException(status_code=404, detail="Module not found or you don't have permission")
    
    module.title = module_data.title
    module.description = module_data.description
    module.order = module_data.order
    
    db.commit()
    db.refresh(module)
    logger.info(f"Module updated successfully: {module_id}")
    return module

# Delete a module
@app.delete("/modules/{module_id}")
def delete_module(
    module_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Deleting module: {module_id} by user: {current_user.email}")
    if not current_user.is_instructor:
        logger.error(f"Non-instructor attempt to delete module: {current_user.email}")
        raise HTTPException(status_code=403, detail="Only instructors can delete modules")
    
    module = db.query(ModuleModel).join(CourseModel).filter(
        ModuleModel.id == module_id,
        CourseModel.instructor_id == current_user.id
    ).first()
    
    if not module:
        logger.error(f"Module not found or permission denied: {module_id} for user: {current_user.id}")
        raise HTTPException(status_code=404, detail="Module not found or you don't have permission")
    
    db.delete(module)
    db.commit()
    logger.info(f"Module deleted successfully: {module_id}")
    return {"message": "Module deleted successfully"}

# Delete a lesson
@app.delete("/lessons/{lesson_id}")
async def delete_lesson(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Deleting lesson: {lesson_id} by user: {current_user.email}")
    if not current_user.is_instructor:
        logger.error(f"Non-instructor attempt to delete lesson: {current_user.email}")
        raise HTTPException(status_code=403, detail="Only instructors can delete lessons")
    
    lesson = db.query(LessonModel).join(ModuleModel).join(CourseModel).filter(
        LessonModel.id == lesson_id,
        CourseModel.instructor_id == current_user.id
    ).first()
    
    if not lesson:
        logger.error(f"Lesson not found or permission denied: {lesson_id} for user: {current_user.id}")
        raise HTTPException(status_code=404, detail="Lesson not found or you don't have permission")
    
    # Remove associated video file from B2 if it exists
    if lesson.video_filename:
        await delete_from_b2(lesson.video_filename)
    
    db.delete(lesson)
    db.commit()
    logger.info(f"Lesson deleted successfully: {lesson_id}")
    return {"message": "Lesson deleted successfully"}

# Progress tracking endpoints
@app.post("/progress/lesson/{lesson_id}/complete")
def mark_lesson_complete(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Marking lesson complete: {lesson_id} by user: {current_user.email}")
    # Find the enrollment for this user and course
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        logger.error(f"Lesson not found: {lesson_id}")
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id
    ).first()
    
    if not enrollment:
        logger.error(f"Not enrolled in course: {lesson.module.course_id} for user: {current_user.id}")
        raise HTTPException(status_code=404, detail="Not enrolled in this course")
    
    # Check if progress already exists
    progress = db.query(ProgressModel).filter(
        ProgressModel.enrollment_id == enrollment.id,
        ProgressModel.lesson_id == lesson_id
    ).first()
    
    if progress:
        # Already completed
        logger.info(f"Lesson already completed: {lesson_id} by user: {current_user.email}")
        return {"message": "Lesson already completed"}
    
    # Create new progress record
    progress = ProgressModel(
        enrollment_id=enrollment.id,
        lesson_id=lesson_id,
        completed=True,
        completed_at=datetime.utcnow()
    )
    db.add(progress)
    db.commit()
    
    logger.info(f"Lesson marked as complete: {lesson_id} by user: {current_user.email}")
    return {"message": "Lesson marked as complete"}

@app.get("/progress/course/{course_id}")
def get_course_progress(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Fetching progress for course: {course_id} by user: {current_user.email}")
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == course_id
    ).first()
    
    if not enrollment:
        logger.error(f"Not enrolled in course: {course_id} for user: {current_user.id}")
        raise HTTPException(status_code=404, detail="Not enrolled in this course")
    
    # Get all lessons in the course
    course = db.query(CourseModel).filter(CourseModel.id == course_id).first()
    total_lessons = 0
    completed_lessons = 0
    
    for module in course.modules:
        total_lessons += len(module.lessons)
        for lesson in module.lessons:
            progress = db.query(ProgressModel).filter(
                ProgressModel.enrollment_id == enrollment.id,
                ProgressModel.lesson_id == lesson.id,
                ProgressModel.completed == True
            ).first()
            if progress:
                completed_lessons += 1
    
    progress_percentage = (completed_lessons / total_lessons * 100) if total_lessons > 0 else 0
    
    logger.info(f"Progress for course {course_id}: {completed_lessons}/{total_lessons} ({progress_percentage:.2f}%)")
    return {
        "total_lessons": total_lessons,
        "completed_lessons": completed_lessons,
        "progress_percentage": progress_percentage
    }

@app.get("/enrollment/check/{course_id}")
def check_enrollment_status(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Checking enrollment for course: {course_id} by user: {current_user.email}")
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == course_id
    ).first()
    
    is_enrolled = enrollment is not None
    logger.info(f"Enrollment status for course {course_id}: {is_enrolled}")
    return {"is_enrolled": is_enrolled}

@app.get("/lessons/{lesson_id}/video-token", response_model=VideoTokenResponse)
async def get_video_token(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Video token request for lesson: {lesson_id} by user: {current_user.email}")
    # Verify user has access to this lesson
    lesson = db.query(LessonModel).filter(LessonModel.id == lesson_id).first()
    if not lesson:
        logger.error(f"Lesson not found: {lesson_id}")
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    # Check if user is enrolled in the course
    enrollment = db.query(EnrollmentModel).filter(
        EnrollmentModel.user_id == current_user.id,
        EnrollmentModel.course_id == lesson.module.course_id
    ).first()
    
    if not enrollment:
        logger.error(f"Not enrolled in course: {lesson.module.course_id} for user: {current_user.id}")
        raise HTTPException(status_code=403, detail="Not enrolled in this course")
    
    # Create a video token
    expires_delta = timedelta(minutes=VIDEO_TOKEN_EXPIRE_MINUTES)
    token_data = {
        "sub": "video_access",
        "user_id": current_user.id,
        "lesson_id": lesson_id
    }
    access_token = create_video_token(token_data, expires_delta=expires_delta)
    
    logger.info(f"Video token generated for lesson: {lesson_id} by user: {current_user.email}")
    return {
        "token": access_token,
        "expires_at": datetime.utcnow() + expires_delta
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
