#!/usr/bin/env python3
import requests
import json

# Test actual POST with very simple data
print('Testing registration with simple test data:')
headers = {
    'Content-Type': 'application/json',
    'Origin': 'http://localhost:5175'
}
data = {
    'email': 'test123@example.com',
    'username': 'test123',
    'password': '1234567890',  # Short password
    'full_name': 'Test User',
    'role': 'student'
}
try:
    response = requests.post('http://127.0.0.1:8000/api/auth/register', json=data, headers=headers)
    print(f'POST Status: {response.status_code}')
    try:
        print(f'Response JSON: {response.json()}')
    except:
        print(f'Response Body: {response.text}')
except Exception as e:
    print(f'Error: {e}')




