#!/usr/bin/env python3
"""
Test script to debug message sending issues
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_send_message():
    """Test the message sending flow"""
    
    print("=" * 70)
    print("MESSAGE SEND TEST")
    print("=" * 70)
    
    # Login as student
    print("\n[1] Logging in as student...")
    login_response = requests.post(f"{BASE_URL}/api/auth/login", json={
        "username": "student",
        "password": "Student123!"
    })
    
    if login_response.status_code != 200:
        print(f"❌ Login failed: {login_response.text}")
        return
    
    student_data = login_response.json()
    student_id = student_data.get('user_id')
    student_token = student_data.get('access_token')
    print(f"✅ Logged in as student (ID: {student_id})")
    print(f"   Token: {student_token[:50]}...")
    
    headers = {"Authorization": f"Bearer {student_token}"}
    
    # Get a counselor
    print("\n[2] Getting available counselors...")
    counselors_response = requests.get(f"{BASE_URL}/api/chat/counselors", headers=headers)
    
    if counselors_response.status_code != 200:
        print(f"❌ Failed to get counselors: {counselors_response.text}")
        return
    
    counselors = counselors_response.json()
    if not counselors:
        print("❌ No counselors available")
        return
    
    counselor = counselors[0]
    counselor_id = counselor['id']
    print(f"✅ Counselor: {counselor['full_name']} (ID: {counselor_id})")
    
    # Try sending a message with detailed error reporting
    print("\n[3] Sending message...")
    message_payload = {
        "receiver_id": counselor_id,
        "message": "Test message from debug script",
        "message_type": "text"
    }
    
    print(f"   Payload: {json.dumps(message_payload, indent=2)}")
    print(f"   Headers: {headers}")
    
    send_response = requests.post(
        f"{BASE_URL}/api/chat/send",
        headers=headers,
        json=message_payload
    )
    
    print(f"\n   Response Status: {send_response.status_code}")
    print(f"   Response Headers: {dict(send_response.headers)}")
    print(f"   Response Body: {send_response.text}")
    
    if send_response.status_code != 200:
        print(f"\n❌ Message send failed!")
        try:
            error_data = send_response.json()
            print(f"   Error details: {json.dumps(error_data, indent=2)}")
        except:
            print(f"   Could not parse error response")
    else:
        print(f"\n✅ Message sent successfully!")
        msg_data = send_response.json()
        print(f"   Message ID: {msg_data.get('id')}")
        print(f"   Content: {msg_data.get('message')}")

if __name__ == "__main__":
    test_send_message()
