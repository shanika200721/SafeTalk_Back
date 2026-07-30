#!/usr/bin/env python3
"""
Test script to verify message persistence in the database
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def log(message):
    """Print with timestamp"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def test_message_persistence():
    """Test the complete message flow"""
    
    log("=" * 60)
    log("MESSAGE PERSISTENCE TEST")
    log("=" * 60)
    
    # Step 1: Login as student
    log("\n[1] Logging in as student...")
    login_response = requests.post(f"{BASE_URL}/api/auth/login", json={
        "username": "student",
        "password": "Student123!"
    })
    
    if login_response.status_code != 200:
        log(f"❌ Login failed: {login_response.text}")
        return False
    
    student_data = login_response.json()
    student_token = student_data.get('access_token')
    student_id = student_data.get('user_id')
    log(f"✅ Logged in as student (ID: {student_id})")
    
    # Step 2: Get available counselors
    log("\n[2] Fetching available counselors...")
    headers = {"Authorization": f"Bearer {student_token}"}
    
    counselors_response = requests.get(
        f"{BASE_URL}/api/chat/counselors",
        headers=headers
    )
    
    if counselors_response.status_code != 200:
        log(f"❌ Failed to get counselors: {counselors_response.text}")
        return False
    
    counselors = counselors_response.json()
    if not counselors:
        log("⚠️  No counselors available")
        return False
    
    counselor = counselors[0]
    counselor_id = counselor['id']
    log(f"✅ Found counselor: {counselor['full_name']} (ID: {counselor_id})")
    
    # Step 3: Send a message
    log("\n[3] Sending a test message...")
    test_message = f"Test message from student at {datetime.now().isoformat()}"
    
    send_response = requests.post(
        f"{BASE_URL}/api/chat/send",
        headers=headers,
        json={
            "receiver_id": counselor_id,
            "message": test_message,
            "message_type": "text"
        }
    )
    
    if send_response.status_code != 200:
        log(f"❌ Failed to send message: {send_response.text}")
        return False
    
    message_data = send_response.json()
    message_id = message_data.get('id')
    log(f"✅ Message sent successfully (ID: {message_id})")
    log(f"   Message content: {message_data.get('message')}")
    
    # Step 4: Retrieve messages
    log("\n[4] Retrieving messages...")
    messages_response = requests.get(
        f"{BASE_URL}/api/chat/messages/{counselor_id}",
        headers=headers,
        params={"limit": 100}
    )
    
    if messages_response.status_code != 200:
        log(f"❌ Failed to retrieve messages: {messages_response.text}")
        return False
    
    messages = messages_response.json()
    log(f"✅ Retrieved {len(messages)} message(s)")
    
    # Step 5: Verify our message is in the list
    log("\n[5] Verifying message persistence...")
    found = False
    for msg in messages:
        if msg.get('id') == message_id:
            found = True
            log(f"✅ Message found in database!")
            log(f"   ID: {msg['id']}")
            log(f"   Sender ID: {msg['sender_id']}")
            log(f"   Receiver ID: {msg['receiver_id']}")
            log(f"   Content: {msg['message']}")
            log(f"   Created at: {msg['created_at']}")
            log(f"   Read: {msg['is_read']}")
            break
    
    if not found:
        log(f"❌ Message was NOT found in retrieved messages!")
        log(f"   Sent message ID: {message_id}")
        log(f"   Retrieved messages: {json.dumps(messages, indent=2)}")
        return False
    
    # Step 6: Check conversations list
    log("\n[6] Checking conversations list...")
    convs_response = requests.get(
        f"{BASE_URL}/api/chat/conversations",
        headers=headers
    )
    
    if convs_response.status_code == 200:
        conversations = convs_response.json()
        log(f"✅ Retrieved {len(conversations)} conversation(s)")
        for conv in conversations:
            if conv['user_id'] == counselor_id:
                log(f"   Conversation with {conv['full_name']}:")
                log(f"   - Last message: {conv['last_message']}")
                log(f"   - Last message time: {conv['last_message_time']}")
                log(f"   - Unread count: {conv['unread_count']}")
    
    log("\n" + "=" * 60)
    log("✅ TEST PASSED - Message persistence is working!")
    log("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = test_message_persistence()
        exit(0 if success else 1)
    except Exception as e:
        log(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
