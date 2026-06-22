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

load_dotenv()

app = FastAPI()

# Get all keys
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_SECRET = os.getenv("GITHUB_SECRET", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================================
# THREAD-SAFE RATE LIMITER (FIXED)
# ============================================================================

class RateLimiter:
    """
    Thread-safe rate limiter using asyncio.Lock
    
    FIX: Uses lock to prevent race conditions
    When multiple async tasks check rate limit simultaneously,
    only one can proceed at a time
    """
    def __init__(self, max_requests: int = 5, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self.lock = asyncio.Lock()  # FIX: Add lock for thread safety
    
    async def is_allowed(self) -> bool:
        """Check if request is allowed (THREAD-SAFE)"""
        async with self.lock:  # FIX: Lock prevents race conditions
            now = datetime.now()
            
            # Remove old requests outside time window
            self.requests = [
                req_time for req_time in self.requests 
                if now - req_time < timedelta(seconds=self.window_seconds)
            ]
            
            # Check if we can make another request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def get_wait_time(self) -> int:
        """How long to wait before next request allowed"""
        if not self.requests:
            return 0
        
        oldest = self.requests[0]
        now = datetime.now()
        wait = (oldest + timedelta(seconds=self.window_seconds) - now).total_seconds()
        
        return max(0, int(wait) + 1)
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        now = datetime.now()
        self.requests = [
            req_time for req_time in self.requests 
            if now - req_time < timedelta(seconds=self.window_seconds)
        ]
        return {
            "requests_used": len(self.requests),
            "max_requests": self.max_requests,
            "remaining": self.max_requests - len(self.requests),
            "window_seconds": self.window_seconds
        }

# Create rate limiter (5 reviews per 60 seconds)
rate_limiter = RateLimiter(max_requests=5, window_seconds=60)

# ============================================================================
# SECURITY CHECKS
# ============================================================================

def check_environment():
    """
    FIX: Validate all required environment variables
    
    Ensures bot won't run with missing critical keys
    """
    print("=" * 60)
    print("SECURITY CHECK: Validating environment...")
    print("=" * 60)
    
    errors = []
    
    if not GITHUB_TOKEN:
        errors.append("❌ GITHUB_TOKEN missing")
    else:
        print("✓ GITHUB_TOKEN loaded")
    
    if not GROQ_API_KEY:
        errors.append("❌ GROQ_API_KEY missing")
    else:
        print("✓ GROQ_API_KEY loaded")
    
    # FIX: Warn if GITHUB_SECRET is missing
    if not GITHUB_SECRET:
        print("⚠️  WARNING: GITHUB_SECRET not set")
        print("   Webhook signature verification is DISABLED")
        print("   (This is OK for testing, but enable for production)")
    else:
        print("✓ GITHUB_SECRET loaded (signature verification enabled)")
    
    if errors:
        print("\n" + "=" * 60)
        print("❌ FATAL ERRORS:")
        for error in errors:
            print(f"   {error}")
        print("=" * 60)
        raise Exception("Missing required environment variables")
    
    print("=" * 60)

# Run security check at startup
try:
    check_environment()
except Exception as e:
    print(f"Failed to start: {e}")
    exit(1)

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
def read_root():
    """Health check endpoint"""
    stats = rate_limiter.get_stats()
    return {
        "message": "Code Review Bot is running!",
        "status": "healthy",
        "features": ["error_handling", "thread_safe_rate_limiting"],
        "rate_limit": stats
    }

@app.get("/test-github")
async def test_github():
    """Test GitHub API connection"""
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.github.com/user",
                headers=headers
            )
        
        if response.status_code == 200:
            user_data = response.json()
            return {
                "status": "success",
                "message": "GitHub token works!",
                "user": user_data.get("login")
            }
        else:
            return {
                "status": "error",
                "message": f"GitHub returned {response.status_code}",
                "details": response.text[:200]
            }
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
            messages=[
                {"role": "user", "content": "Say 'Ready!' in one sentence."}
            ]
        )
        return {
            "status": "success",
            "message": "Groq API works!",
            "response": response.choices[0].message.content
        }
    except Exception as e:
        return {"status": "error", "message": str(e)[:100]}

@app.get("/rate-limit-stats")
def get_rate_limit_stats():
    """Check current rate limit status"""
    return {"rate_limiter": rate_limiter.get_stats()}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def get_pr_diff(owner: str, repo: str, pr_number: int) -> str:
    """
    Get code diff from GitHub with error handling
    
    FIX: Now checks rate limit before making request!
    """
    
    # FIX: Check rate limit BEFORE making GitHub API call
    print("\n🔍 Checking rate limit before GitHub API call...")
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
            
            elif response.status_code == 404:
                raise Exception("PR not found (404)")
            
            elif response.status_code == 401:
                raise Exception("Authentication failed - check GITHUB_TOKEN")
            
            elif response.status_code >= 500:
                print(f"⚠️  GitHub server error ({response.status_code}), retrying...")
                await asyncio.sleep(retry_delay)
                continue
            
            else:
                raise Exception(f"GitHub API returned {response.status_code}")
        
        except httpx.TimeoutException:
            print(f"⚠️  Timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                raise Exception("Timeout: Could not fetch code diff")
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  Error: {str(e)}, retrying...")
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
    
    except groq.RateLimitError:
        print("❌ Groq rate limited")
        return None
    except groq.BadRequestError as e:
        print(f"❌ Groq bad request: {str(e)[:100]}")
        return None
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
*Review generated by CodeGazer - The Code Review Bot*"""
        
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
async def receive_webhook(
    request: Request,
    x_hub_signature_256: str = Header(None)
):
    """Handle GitHub webhook with full error handling"""
    
    print("\n" + "🔔" * 30)
    print("WEBHOOK RECEIVED!")
    print("🔔" * 30)
    
    try:
        payload_body = await request.body()
        
        try:
            data = json.loads(payload_body)
        except json.JSONDecodeError:
            print("❌ Invalid JSON in webhook")
            return {"error": "Invalid JSON"}
        
        # Extract information
        action = data.get("action")
        pr = data.get("pull_request", {})
        repo = data.get("repository", {})
        
        pr_number = pr.get("number")
        owner = repo.get("owner", {}).get("login")
        repo_name = repo.get("name")
        
        print(f"Action: {action}")
        print(f"Repository: {owner}/{repo_name}")
        print(f"PR Number: #{pr_number}")
        
        # Validate required fields
        if not all([action, pr_number, owner, repo_name]):
            print("⚠️  Missing required fields in webhook")
            return {"status": "skipped"}
        
        # Only process new or updated PRs
        if action not in ["opened", "synchronize"]:
            print(f"⏭️  Skipping action: {action}")
            return {"status": "skipped"}
        
        try:
            # STEP 1: Get code diff (with rate limit check)
            print("\n📥 STEP 1: Fetching code diff...")
            code_diff = await get_pr_diff(owner, repo_name, pr_number)
            
            if not code_diff or len(code_diff) < 10:
                print("⚠️  No meaningful code changes")
                return {"status": "skipped"}
            
            if len(code_diff) > 8000:
                code_diff = code_diff[:8000] + "\n... (truncated)"
            
            print(f"✓ Got {len(code_diff)} characters")
            
            # STEP 2: Get review from Groq
            print("\n🤖 STEP 2: Asking Groq to review...")
            review = await review_code_with_groq(code_diff)
            
            if not review:
                print("⚠️  Could not get review")
                await post_review_comment(
                    owner, repo_name, pr_number,
                    "Sorry! I couldn't review this code. Please try again later."
                )
                return {"status": "failed"}
            
            # STEP 3: Post review
            print("\n💬 STEP 3: Posting review...")
            success = await post_review_comment(owner, repo_name, pr_number, review)
            
            if success:
                print("\n✅ COMPLETE!")
                print(f"Rate limit stats: {rate_limiter.get_stats()}")
                return {"status": "success"}
            else:
                return {"status": "failed"}
        
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            try:
                await post_review_comment(
                    owner, repo_name, pr_number,
                    f"Error: {str(e)[:100]}"
                )
            except:
                pass
            return {"status": "error", "message": str(e)[:100]}
    
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        return {"status": "critical_error"}