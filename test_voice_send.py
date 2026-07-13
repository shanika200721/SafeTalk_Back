#!/usr/bin/env python
"""Test voice message sending endpoint"""

import requests
import json
import io

API_URL = "http://localhost:8000/api"

# Login first - use working credentials
login_data = {"username": "student123", "password": "StudentPass123!"}

print("📝 Attempting login with:", login_data['username'])
login_response = requests.post(
    f"{API_URL}/auth/login",
    json=login_data
)

print(f"Login Status: {login_response.status_code}")
if login_response.status_code != 200:
    print("❌ Login failed!")
    print(json.dumps(login_response.json(), indent=2))
    exit(1)

token = login_response.json().get('access_token')
print(f"✅ Login successful, token: {token[:20]}...")

headers = {
    "Authorization": f"Bearer {token}"
}

# Create a simple test audio file (WAV header + minimal data)
# This is a minimal valid WAV file for testing
wav_header = bytes([
    0x52, 0x49, 0x46, 0x46,  # "RIFF"
    0x24, 0x00, 0x00, 0x00,  # File size
    0x57, 0x41, 0x56, 0x45,  # "WAVE"
    0x66, 0x6D, 0x74, 0x20,  # "fmt "
    0x10, 0x00, 0x00, 0x00,  # Subchunk1Size
    0x01, 0x00,              # AudioFormat (PCM)
    0x01, 0x00,              # NumChannels
    0x44, 0xAC, 0x00, 0x00,  # SampleRate (44100)
    0x88, 0x58, 0x01, 0x00,  # ByteRate
    0x02, 0x00,              # BlockAlign
    0x10, 0x00,              # BitsPerSample
    0x64, 0x61, 0x74, 0x61,  # "data"
    0x00, 0x00, 0x00, 0x00   # Subchunk2Size
])

print("\n📤 Preparing to send voice message...")
print(f"Audio file size: {len(wav_header)} bytes")

# Prepare multipart form data
files = {
    'audio': ('voice_message.wav', io.BytesIO(wav_header), 'audio/wav')
}
data = {
    'receiver_id': '2'  # Try sending to user ID 2
}

print("\n📨 Request details:")
print(f"  URL: {API_URL}/chat/send-voice")
print(f"  Method: POST")
print(f"  Headers: {headers}")
print(f"  Data: {data}")
print(f"  File: voice_message.wav ({len(wav_header)} bytes)")

try:
    response = requests.post(
        f"{API_URL}/chat/send-voice",
        files=files,
        data=data,
        headers=headers,
        timeout=10
    )

    print(f"\n📊 Response Status: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Body: {json.dumps(response.json(), indent=2)}")

    if response.status_code == 200:
        print("✅ Voice message sent successfully!")
    else:
        print("❌ Failed to send voice message")
        
except requests.exceptions.RequestException as e:
    print(f"❌ Request failed: {e}")
    print(f"Exception type: {type(e).__name__}")
