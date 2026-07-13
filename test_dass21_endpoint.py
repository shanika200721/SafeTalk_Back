#!/usr/bin/env python
"""
Test DASS21 endpoint
Run this to verify the endpoint works
"""

import requests
import json

BASE_URL = "http://localhost:8000"

# Test data - must be exactly 21 responses (0-3 scale)
test_responses = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1]

print("=" * 60)
print("DASS21 Assessment API Test")
print("=" * 60)

# First, login to get auth token
print("\n1️⃣ Logging in...")
login_data = {
    "username": "teststudent",
    "password": "Password123!"
}

login_response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
if login_response.status_code != 200:
    print(f"❌ Login failed: {login_response.status_code}")
    print(login_response.json())
    exit(1)

token = login_response.json().get("access_token")
print(f"✅ Login successful! Token: {token[:20]}...")

# Test DASS21 endpoint
print("\n2️⃣ Testing DASS21 endpoint...")
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

payload = {
    "responses": test_responses
}

print(f"Sending payload: {json.dumps(payload, indent=2)}")

response = requests.post(
    f"{BASE_URL}/api/assessments/dass21",
    json=payload,
    headers=headers
)

print(f"\nStatus Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

if response.status_code == 200:
    print("\n✅ DASS21 Assessment submitted successfully!")
    data = response.json()
    print(f"Depression Score: {data.get('depression_score')}")
    print(f"Anxiety Score: {data.get('anxiety_score')}")
    print(f"Stress Score: {data.get('stress_score')}")
    print(f"Total Score: {data.get('total_dass21_score')}")
else:
    print("\n❌ Failed to submit DASS21 Assessment")
    print(f"Error: {response.json()}")
