from fastapi import FastAPI, Request, Header
import os
from dotenv import load_dotenv
import hmac
import hashlib
import json
import httpx
import groq
from datetime import datetime, timedelta
import time
from typing import Optional
import asyncio

load_dotenv()

app = FastAPI()

# Get all keys
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_SECRET = os.getenv("GITHUB_SECRET", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Rate limiting
class RateLimiter:
    """
    Controls how many reviews we do per minute
    
    Analogy: Like a bouncer at a club
    Only lets X people in per minute
    """
    def __init__(self, max_requests: int = 5, window_seconds: int = 60):
        self.max_requests = max_requests  # Max 5 requests
        self.window_seconds = window_seconds  # Per 60 seconds
        self.requests = []  # Track times of requests
    
    async def is_allowed(self) -> bool:
        """Check if we can make a request"""
        now = datetime.now()
        
        # Remove old requests (outside time window)
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

# Create rate limiter (5 reviews per 60 seconds)
rate_limiter = RateLimiter(max_requests=5, window_seconds=60)

# Verify keys are loaded
print("=" * 50)
print("CHECKING KEYS...")
print("=" * 50)
print(f"✓ GITHUB_TOKEN: {'Loaded ✓' if GITHUB_TOKEN else 'MISSING ✗'}")
print(f"✓ GROQ_API_KEY: {'Loaded ✓' if GROQ_API_KEY else 'MISSING ✗'}")
print("=" * 50)

@app.get("/")
def read_root():
    return {
        "message": "Code Review Bot is running!",
        "status": "healthy",
        "features": ["error_handling", "rate_limiting"]
    }

@app.get("/test-github")
async def test_github():
    """Test if GitHub token works"""
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
    except httpx.TimeoutException:
        return {"status": "error", "message": "GitHub API timeout (too slow)"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/test-groq")
def test_groq():
    """Test if Groq API key works"""
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Say 'Groq is ready!' in one sentence."}
            ]
        )
        return {
            "status": "success",
            "message": "Groq API works!",
            "response": response.choices[0].message.content
        }
    except groq.BadRequestError as e:
        # Model decommissioned
        return {
            "status": "error",
            "message": f"Model error: {str(e)[:100]}",
            "hint": "Update model name in main.py"
        }
    except groq.RateLimitError:
        return {
            "status": "error",
            "message": "Groq rate limit exceeded"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

async def get_pr_diff(owner: str, repo: str, pr_number: int) -> str:
    """
    Get code diff from GitHub with error handling
    
    Retries up to 3 times if it fails
    """
    max_retries = 3
    retry_delay = 2  # seconds
    
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
                raise Exception("Authentication failed - check GitHub token")
            
            elif response.status_code >= 500:
                # Server error - try again
                print(f"⚠️  GitHub server error ({response.status_code}), retrying...")
                await asyncio.sleep(retry_delay)
                continue
            
            else:
                raise Exception(f"GitHub API returned {response.status_code}")
        
        except httpx.TimeoutException:
            print(f"⚠️  Timeout fetching diff (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                raise Exception("Timeout: Could not fetch code diff after 3 attempts")
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  Error: {str(e)}, retrying...")
                await asyncio.sleep(retry_delay)
            else:
                raise

async def review_code_with_groq(code_diff: str) -> Optional[str]:
    """
    Send code to Groq with error handling
    
    Returns review text or None if fails
    """
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        
        # Shorten prompt to save tokens
        prompt = f"""Review this code diff. Find 3-5 issues:

```diff
{code_diff}
```

Focus: bugs, security, performance. Be brief."""
        
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            max_tokens=800,
            messages=[
                {"role": "user", "content": prompt}
            ],
            timeout=30.0  # 30 second timeout
        )
        
        review = response.choices[0].message.content
        print(f"✓ Got review from Groq ({len(review)} chars)")
        return review
    
    except groq.RateLimitError:
        print("❌ Groq rate limited - too many requests")
        return None
    
    except groq.BadRequestError as e:
        error_msg = str(e)
        if "model" in error_msg.lower() and "decommissioned" in error_msg.lower():
            print("❌ Model decommissioned - update model name")
        else:
            print(f"❌ Bad request: {error_msg[:100]}")
        return None
    
    except groq.APIConnectionError:
        print("❌ Cannot connect to Groq API")
        return None
    
    except Exception as e:
        print(f"❌ Groq error: {str(e)[:100]}")
        return None

async def post_review_comment(owner: str, repo: str, pr_number: int, review: str) -> bool:
    """
    Post review with error handling
    
    Returns True if successful
    """
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        comment_body = f"""🤖 **Automated Code Review**

{review}

---
*This review was generated by Code Review Bot*"""
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments",
                headers=headers,
                json={"body": comment_body}
            )
        
        if response.status_code == 201:
            print(f"✓ Review posted successfully!")
            return True
        
        elif response.status_code == 401:
            print("❌ Authentication failed - check GitHub token")
            return False
        
        elif response.status_code == 403:
            print("❌ Permission denied - bot can't post comments")
            return False
        
        elif response.status_code == 404:
            print("❌ PR not found")
            return False
        
        else:
            print(f"❌ Failed to post ({response.status_code}): {response.text[:100]}")
            return False
    
    except httpx.TimeoutException:
        print("❌ Timeout posting review")
        return False
    
    except Exception as e:
        print(f"❌ Error posting review: {str(e)[:100]}")
        return False

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
            return {"error": "Invalid JSON", "status": "failed"}
        
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
            return {"status": "skipped", "reason": "Invalid webhook data"}
        
        # Only process new or updated PRs
        if action not in ["opened", "synchronize"]:
            print(f"⏭️  Skipping action: {action}")
            return {"status": "skipped"}
        
        # CHECK RATE LIMIT
        print("\n⏱️  Checking rate limit...")
        if not await rate_limiter.is_allowed():
            wait_time = rate_limiter.get_wait_time()
            print(f"⚠️  Rate limited! Wait {wait_time} seconds")
            return {
                "status": "rate_limited",
                "message": f"Please wait {wait_time} seconds before next review",
                "wait_seconds": wait_time
            }
        
        print("✓ Rate limit OK (within limits)")
        
        try:
            # STEP 1: Get code diff
            print("\n📥 STEP 1: Fetching code diff...")
            code_diff = await get_pr_diff(owner, repo_name, pr_number)
            
            if not code_diff or len(code_diff) < 10:
                print("⚠️  No meaningful code changes to review")
                return {
                    "status": "skipped",
                    "reason": "No code changes found"
                }
            
            # Limit size
            if len(code_diff) > 8000:
                code_diff = code_diff[:8000] + "\n... (truncated)"
            
            print(f"✓ Got {len(code_diff)} characters")
            
            # STEP 2: Get review from Groq
            print("\n🤖 STEP 2: Asking Groq to review...")
            review = await review_code_with_groq(code_diff)
            
            if not review:
                print("⚠️  Could not get review from Groq")
                # Post a comment saying we failed
                await post_review_comment(
                    owner, repo_name, pr_number,
                    "Sorry! I couldn't review this code right now. Please try again later."
                )
                return {
                    "status": "failed",
                    "message": "Groq API failed"
                }
            
            # STEP 3: Post review
            print("\n💬 STEP 3: Posting review...")
            success = await post_review_comment(owner, repo_name, pr_number, review)
            
            if success:
                print("\n✅ COMPLETE!")
                return {
                    "status": "success",
                    "message": "Review posted",
                    "pr": f"{owner}/{repo_name}#{pr_number}"
                }
            else:
                return {
                    "status": "failed",
                    "message": "Could not post review"
                }
        
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            
            # Try to post error message on PR
            try:
                await post_review_comment(
                    owner, repo_name, pr_number,
                    f"Error reviewing code: {str(e)[:100]}\n\nPlease check the bot logs."
                )
            except:
                pass
            
            return {
                "status": "error",
                "message": str(e)[:100]
            }
    
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        return {
            "status": "critical_error",
            "message": str(e)[:100]
        }