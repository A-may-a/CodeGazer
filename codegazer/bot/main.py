from fastapi import FastAPI, Request, Header
import os
from dotenv import load_dotenv
import hmac
import hashlib
import json
import httpx
import groq
from datetime import datetime, timedelta
import asyncio
from typing import Optional
import asyncio

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

load_dotenv()

# ============================================================================
# DATABASE SETUP (PostgreSQL)
# ============================================================================

# Get database URL from .env
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ ERROR: DATABASE_URL not set in .env file")
    print("Example: postgresql://user:password@localhost:5432/codegazer_bot")
    exit(1)

print(f"✓ Connecting to database: {DATABASE_URL.split('@')[1]}")

# Create engine (connection to PostgreSQL)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True  # Test connection before using
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# ============================================================================
# DATABASE MODELS (Tables)
# ============================================================================

class Review(Base):
    """
    Represents a code review stored in PostgreSQL database
    
    Columns:
    - id: Unique ID for each review
    - repository: Repository name (e.g., "codegazer-test")
    - owner: GitHub username (e.g., "A-may-a")
    - pr_number: PR number (e.g., 1)
    - code_diff: Code that was reviewed
    - review_text: Groq's feedback
    - status: "success", "failed", "rate_limited"
    - created_at: When review was done
    - error_message: Error details if failed
    """
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    owner = Column(String, index=True, nullable=False)
    repository = Column(String, index=True, nullable=False)
    pr_number = Column(Integer, index=True, nullable=False)
    code_diff = Column(Text, nullable=False)
    review_text = Column(Text, nullable=False)
    status = Column(String, default="pending")  # success, failed, rate_limited
    created_at = Column(DateTime, default=datetime.utcnow)
    error_message = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<Review #{self.id}: {self.owner}/{self.repository}#{self.pr_number} ({self.status})>"

# Create all tables
try:
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created/verified")
except Exception as e:
    print(f"❌ Error creating tables: {e}")
    print("Make sure PostgreSQL is running and credentials are correct")
    exit(1)

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_review(
    db: Session,
    owner: str,
    repository: str,
    pr_number: int,
    code_diff: str,
    review_text: str,
    status: str = "success",
    error_message: str = None
) -> Review:
    """
    Save a review to PostgreSQL database
    
    Example:
    review = save_review(
        db,
        owner="A-may-a",
        repository="codegazer-test",
        pr_number=1,
        code_diff="def test(): pass",
        review_text="Looks good!",
        status="success"
    )
    """
    try:
        db_review = Review(
            owner=owner,
            repository=repository,
            pr_number=pr_number,
            code_diff=code_diff,
            review_text=review_text,
            status=status,
            error_message=error_message,
            created_at=datetime.utcnow()
        )
        
        db.add(db_review)
        db.commit()
        db.refresh(db_review)
        
        print(f"✓ Saved review #{db_review.id} to PostgreSQL")
        return db_review
    
    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        db.rollback()
        raise

def get_reviews(db: Session, owner: str = None, repository: str = None, limit: int = 100):
    """
    Get reviews from PostgreSQL database
    
    Can filter by owner and/or repository
    """
    try:
        query = db.query(Review)
        
        if owner:
            query = query.filter(Review.owner == owner)
        
        if repository:
            query = query.filter(Review.repository == repository)
        
        return query.order_by(Review.created_at.desc()).limit(limit).all()
    
    except Exception as e:
        print(f"❌ Error querying database: {e}")
        return []

def get_review_stats(db: Session):
    """Get statistics about reviews"""
    try:
        all_reviews = db.query(Review).all()
        
        if not all_reviews:
            return {
                "total_reviews": 0,
                "successful": 0,
                "failed": 0,
                "rate_limited": 0,
                "success_rate": "0%",
                "average_review_length": 0
            }
        
        successful = len([r for r in all_reviews if r.status == "success"])
        failed = len([r for r in all_reviews if r.status == "failed"])
        rate_limited = len([r for r in all_reviews if r.status == "rate_limited"])
        
        avg_length = sum(len(r.review_text) for r in all_reviews) / len(all_reviews) if all_reviews else 0
        
        return {
            "total_reviews": len(all_reviews),
            "successful": successful,
            "failed": failed,
            "rate_limited": rate_limited,
            "success_rate": f"{(successful / len(all_reviews) * 100):.1f}%",
            "average_review_length": int(avg_length),
            "database": "PostgreSQL"
        }
    
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return {"error": str(e)}

# ============================================================================
# FASTAPI SETUP
# ============================================================================

app = FastAPI()

# Get all keys
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_SECRET = os.getenv("GITHUB_SECRET", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Rate limiter
class RateLimiter:
    def __init__(self, max_requests: int = 5, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def is_allowed(self) -> bool:
        async with self.lock:
            now = datetime.now()
            self.requests = [
                req_time for req_time in self.requests 
                if now - req_time < timedelta(seconds=self.window_seconds)
            ]
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
    
    def get_wait_time(self) -> int:
        if not self.requests:
            return 0
        oldest = self.requests[0]
        now = datetime.now()
        wait = (oldest + timedelta(seconds=self.window_seconds) - now).total_seconds()
        return max(0, int(wait) + 1)
    
    def get_stats(self) -> dict:
        now = datetime.now()
        self.requests = [
            req_time for req_time in self.requests 
            if now - req_time < timedelta(seconds=self.window_seconds)
        ]
        return {
            "requests_used": len(self.requests),
            "max_requests": self.max_requests,
            "remaining": self.max_requests - len(self.requests)
        }

rate_limiter = RateLimiter(max_requests=5, window_seconds=60)

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
def read_root():
    """Health check with database stats"""
    db = SessionLocal()
    try:
        stats = get_review_stats(db)
        return {
            "message": "Code Review Bot is running!",
            "status": "healthy",
            "database": "PostgreSQL",
            "review_stats": stats
        }
    finally:
        db.close()

@app.get("/reviews")
def get_all_reviews(owner: str = None, repository: str = None):
    """
    Get stored reviews from PostgreSQL
    
    Usage:
    - /reviews
    - /reviews?owner=A-may-a
    - /reviews?repository=codegazer-test
    - /reviews?owner=A-may-a&repository=codegazer-test
    """
    db = SessionLocal()
    try:
        reviews = get_reviews(db, owner=owner, repository=repository)
        
        return {
            "database": "PostgreSQL",
            "count": len(reviews),
            "reviews": [
                {
                    "id": r.id,
                    "owner": r.owner,
                    "repository": r.repository,
                    "pr_number": r.pr_number,
                    "status": r.status,
                    "review": r.review_text[:200] + "..." if len(r.review_text) > 200 else r.review_text,
                    "created_at": r.created_at.isoformat(),
                    "error": r.error_message
                }
                for r in reviews
            ]
        }
    finally:
        db.close()

@app.get("/stats")
def get_stats():
    """Get detailed review statistics"""
    db = SessionLocal()
    try:
        stats = get_review_stats(db)
        return {"stats": stats}
    finally:
        db.close()

@app.get("/test-github")
async def test_github():
    """Test GitHub API connection"""
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("https://api.github.com/user", headers=headers)
        if response.status_code == 200:
            user_data = response.json()
            return {"status": "success", "user": user_data.get("login")}
        else:
            return {"status": "error", "message": f"GitHub returned {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/test-groq")
def test_groq():
    """Test Groq API connection"""
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'Ready!' in one sentence."}]
        )
        return {"status": "success", "message": response.choices[0].message.content}
    except Exception as e:
        return {"status": "error", "message": str(e)[:100]}

@app.get("/test-database")
def test_database():
    """Test PostgreSQL connection"""
    try:
        db = SessionLocal()
        # Try to query
        count = db.query(Review).count()
        db.close()
        return {
            "status": "success",
            "message": "PostgreSQL connected!",
            "total_reviews": count
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"PostgreSQL connection failed: {str(e)}"
        }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def get_pr_diff(owner: str, repo: str, pr_number: int) -> str:
    """Get code diff from GitHub"""
    print("\n🔍 Checking rate limit...")
    if not await rate_limiter.is_allowed():
        wait_time = rate_limiter.get_wait_time()
        raise Exception(f"Rate limited! Wait {wait_time} seconds")
    
    max_retries = 3
    retry_delay = 2
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.diff"
    }
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}",
                    headers=headers
                )
            
            if response.status_code == 200:
                print(f"✓ Got code diff (attempt {attempt + 1})")
                return response.text
            elif response.status_code >= 500:
                print(f"⚠️  Server error, retrying...")
                await asyncio.sleep(retry_delay)
            else:
                raise Exception(f"GitHub API returned {response.status_code}")
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                raise

async def review_code_with_groq(code_diff: str) -> Optional[str]:
    """Send code to Groq for review"""
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        prompt = f"""Review this code diff. Find 3-5 issues:

```diff
{code_diff}
```

Focus: bugs, security, performance. Be brief."""
        
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
            timeout=30.0
        )
        review = response.choices[0].message.content
        print(f"✓ Got review from Groq ({len(review)} chars)")
        return review
    except Exception as e:
        print(f"❌ Groq error: {str(e)[:100]}")
        return None

async def post_review_comment(owner: str, repo: str, pr_number: int, review: str) -> bool:
    """Post review as comment on PR"""
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        comment_body = f"""🤖 **Automated Code Review**

{review}

---
*Review generated by Code Review Bot*"""
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments",
                headers=headers,
                json={"body": comment_body}
            )
        
        if response.status_code == 201:
            print(f"✓ Review posted successfully!")
            return True
        else:
            print(f"❌ Failed to post ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Error posting review: {str(e)[:100]}")
        return False

# ============================================================================
# MAIN WEBHOOK HANDLER
# ============================================================================

@app.post("/webhook")
async def receive_webhook(request: Request, x_hub_signature_256: str = Header(None)):
    """Handle GitHub webhook and save to PostgreSQL"""
    
    print("\n" + "🔔" * 30)
    print("WEBHOOK RECEIVED!")
    print("🔔" * 30)
    
    db = SessionLocal()
    
    try:
        payload_body = await request.body()
        
        try:
            data = json.loads(payload_body)
        except json.JSONDecodeError:
            print("❌ Invalid JSON")
            db.close()
            return {"error": "Invalid JSON"}
        
        action = data.get("action")
        pr = data.get("pull_request", {})
        repo = data.get("repository", {})
        
        pr_number = pr.get("number")
        owner = repo.get("owner", {}).get("login")
        repo_name = repo.get("name")
        
        print(f"Action: {action}")
        print(f"Repository: {owner}/{repo_name}")
        print(f"PR Number: #{pr_number}")
        
        if not all([action, pr_number, owner, repo_name]):
            print("⚠️  Missing required fields")
            db.close()
            return {"status": "skipped"}
        
        if action not in ["opened", "synchronize"]:
            print(f"⏭️  Skipping action: {action}")
            db.close()
            return {"status": "skipped"}
        
        try:
            print("\n📥 STEP 1: Fetching code diff...")
            code_diff = await get_pr_diff(owner, repo_name, pr_number)
            
            if not code_diff or len(code_diff) < 10:
                print("⚠️  No code changes")
                db.close()
                return {"status": "skipped"}
            
            if len(code_diff) > 8000:
                code_diff = code_diff[:8000] + "\n... (truncated)"
            
            print(f"✓ Got {len(code_diff)} characters")
            
            print("\n🤖 STEP 2: Asking Groq to review...")
            review = await review_code_with_groq(code_diff)
            
            if not review:
                print("⚠️  Could not get review")
                save_review(
                    db, owner, repo_name, pr_number, code_diff,
                    "Failed to get review", status="failed",
                    error_message="Groq API error"
                )
                await post_review_comment(owner, repo_name, pr_number,
                    "Sorry! I couldn't review this code. Please try again later.")
                db.close()
                return {"status": "failed"}
            
            print("\n💬 STEP 3: Posting review...")
            success = await post_review_comment(owner, repo_name, pr_number, review)
            
            if success:
                # SAVE TO PostgreSQL ✓
                save_review(
                    db, owner, repo_name, pr_number, code_diff,
                    review, status="success"
                )
                print("\n✅ COMPLETE!")
                db.close()
                return {"status": "success", "database": "PostgreSQL"}
            else:
                save_review(
                    db, owner, repo_name, pr_number, code_diff,
                    review, status="failed",
                    error_message="Failed to post comment"
                )
                db.close()
                return {"status": "failed"}
        
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            save_review(
                db, owner, repo_name, pr_number, "",
                "", status="failed",
                error_message=str(e)[:100]
            )
            db.close()
            return {"status": "error", "message": str(e)[:100]}
    
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        db.close()
        return {"status": "critical_error"}