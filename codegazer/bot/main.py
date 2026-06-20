from fastapi import FastAPI, Request, Header
import os
from dotenv import load_dotenv
import hmac
import hashlib
import json

load_dotenv()

app = FastAPI()

GITHUB_SECRET = os.getenv("GITHUB_SECRET", "")

@app.get("/")
def read_root():
    return {
        "message": "Code Review Bot is running!",
        "status": "healthy"
    }

# Function to verify the webhook is from GitHub (security)
def verify_github_signature(payload_body, signature):
    """
    This checks if the message really came from GitHub.
    
    """
    if not signature:
        return False
    
    # Create a signature
    hash_object = hmac.new(
        GITHUB_SECRET.encode(),
        payload_body,
        hashlib.sha256
    )
    expected_signature = "sha256=" + hash_object.hexdigest()
    
    # Compare
    return hmac.compare_digest(expected_signature, signature)

@app.post("/webhook")
async def receive_webhook(
    request: Request,
    x_hub_signature_256: str = Header(None)
):
    """
    This is called when GitHub sends a webhook.
    
    Analogy:
    - Doorbell rings = GitHub sends webhook
    - You check who's at door = We verify signature
    - You open door = We process the data
    """
    
    # Get the raw data
    payload_body = await request.body()
    
    # Verify it's really from GitHub
    if not verify_github_signature(payload_body, x_hub_signature_256):
        return {
            "error": "Invalid signature",
            "status": "rejected"
        }
    
    # Convert to Python dictionary
    data = json.loads(payload_body)
    
    # Extract useful information
    action = data.get("action")  # "opened", "synchronize", "closed"
    pr_number = data.get("pull_request", {}).get("number")
    repo_name = data.get("repository", {}).get("name")
    repo_owner = data.get("repository", {}).get("owner", {}).get("login")
    
    print(f"✓ Webhook received!")
    print(f"  Action: {action}")
    print(f"  PR Number: {pr_number}")
    print(f"  Repository: {repo_owner}/{repo_name}")
    
    # For  just acknowledge
    return {
        "message": "Webhook processed",
        "status": "success",
        "pr_number": pr_number
    }

#