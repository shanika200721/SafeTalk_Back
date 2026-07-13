#!/usr/bin/env python3
"""
Diagnostic script to understand message retrieval issues
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def diagnose():
    """Diagnose message retrieval issues"""
    
    print("=" * 70)
    print("MESSAGE RETRIEVAL DIAGNOSTIC")
    print("=" * 70)
    
    # Login as student
    print("\n[1] Logging in as student...")
    login_response = requests.post(f"{BASE_URL}/api/auth/login", json={
        "username": "student",
        "password": "Student123!"
    })
    
    student_data = login_response.json()
    student_id = student_data.get('user_id')
    student_token = student_data.get('access_token')
    print(f"✅ Student ID: {student_id}")
    
    headers = {"Authorization": f"Bearer {student_token}"}
    
    # Get conversations
    print("\n[2] Getting conversations...")
    convs_response = requests.get(f"{BASE_URL}/api/chat/conversations", headers=headers)
    conversations = convs_response.json()
    print(f"Total conversations: {len(conversations)}")
    
    for conv in conversations:
        print(f"\n   Conversation with {conv['full_name']} (ID: {conv['user_id']}):")
        print(f"   - Last message: {conv['last_message']}")
        print(f"   - Last message time: {conv['last_message_time']}")
        print(f"   - Unread count: {conv['unread_count']}")
        
        # Get messages for this conversation
        print(f"\n   Fetching messages with user {conv['user_id']}...")
        msgs_response = requests.get(
            f"{BASE_URL}/api/chat/messages/{conv['user_id']}",
            headers=headers,
            params={"limit": 50}
        )
        
        if msgs_response.status_code == 200:
            messages = msgs_response.json()
            print(f"   Retrieved {len(messages)} messages:")
            for msg in messages[-5:]:  # Show last 5 messages
                sender_type = "Student" if msg['sender_id'] == student_id else "Counselor"
                print(f"      [{msg['id']}] {sender_type}: {msg['message'][:60]}... ({msg['created_at']})")
        else:
            print(f"   Error fetching messages: {msgs_response.text}")
    
    # Get available counselors
    print("\n[3] Getting available counselors...")
    counselors_response = requests.get(f"{BASE_URL}/api/chat/counselors", headers=headers)
    counselors = counselors_response.json()
    print(f"Total available counselors: {len(counselors)}")
    
    for counselor in counselors:
        print(f"\n   Counselor: {counselor['full_name']} (ID: {counselor['id']})")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    diagnose()
