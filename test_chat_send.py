#!/usr/bin/env python
"""Test chat send endpoint directly"""

import requests
import json

API_URL = "http://localhost:8000/api"

# Login first to get token
login_response = requests.post(
    f"{API_URL}/auth/login",
    json={"username": "student123", "password": "StudentPass123!"}
)

if login_response.status_code != 200:
    print(f"❌ Login failed: {login_response.status_code}")
    print(login_response.json())
    exit(1)

token = login_response.json().get('access_token')
print(f"✅ Login successful, token: {token[:20]}...")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Try to send message
payload = {
    "receiver_id": 2,
    "message": "Test message from script",
    "message_type": "text"
}

print(f"\n📨 Sending message with payload: {json.dumps(payload, indent=2)}")

response = requests.post(
    f"{API_URL}/chat/send",
    json=payload,
    headers=headers
)

print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

if response.status_code == 200:
    print("✅ Message sent successfully!")
else:
    print("❌ Failed to send message")
