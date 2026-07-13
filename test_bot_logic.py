#!/usr/bin/env python
"""Direct bot logic test - test crisis detection and LLM usage without API"""

import sys
sys.path.insert(0, 'c:\\Users\\CMSDEV04\\Documents\\research\\suicide-prevention-agent\\backend')

from app.ml.counselor_safetalk_bot import CounselorSafeTalkBot
from app.ml.mental_health_knowledge_base import MentalHealthKnowledgeBase
import json

def test_bot_crisis_logic():
    """Test the bot's crisis detection directly"""
    
    print("=" * 80)
    print("DIRECT BOT CRISIS DETECTION TEST")
    print("=" * 80)
    
    # Initialize bot and KB
    bot = CounselorSafeTalkBot()
    
    test_cases = [
        ("hi", "greeting", False),
        ("i feel sad", "depression", False),
        ("i want to fix my sadness mood", "depression", False),
        ("i've been lonely for months", "loneliness", False),
        ("i want to kill myself", "suicidal_ideation", True),
        ("i'm planning to end it all", "suicidal_ideation", True),
        ("i hate myself", "low_self_worth", False),
    ]
    
    for message, expected_intent, should_have_crisis in test_cases:
        print(f"\n{'='*80}")
        print(f"Testing: '{message}'")
        print(f"Expected intent: {expected_intent}")
        print(f"Should have crisis: {should_have_crisis}")
        
        # Get bot response
        response = bot.generate_response(message, user_id=None)
        
        # Check crisis detection
        crisis_by_keywords = response.get('is_crisis', False)
        crisis_level = response.get('crisis_level', 0)
        intent = response.get('intent', 'unknown')
        llm_used = response.get('llm_used', False)
        
        print(f"\nResults:")
        print(f"  Intent detected: {intent}")
        print(f"  Crisis by keywords: {crisis_by_keywords}")
        print(f"  Crisis level: {crisis_level}/10")
        print(f"  LLM used: {llm_used}")
        
        # Simulate the API endpoint logic (lines 96-102 of bot.py)
        is_crisis_from_keywords = response.get("is_crisis", False)
        is_crisis_api = is_crisis_from_keywords or crisis_level >= 9
        
        print(f"  API would set is_crisis: {is_crisis_api}")
        
        # Check response
        response_text = response.get('main_response', '')[:100]
        print(f"  Response: {response_text}...")
        
        # Verify
        if crisis_by_keywords == should_have_crisis:
            print(f"\n✅ PASS: Crisis detection correct for this message")
        else:
            print(f"\n❌ FAIL: Crisis detection wrong! Expected {should_have_crisis}, got {crisis_by_keywords}")
        
        # Check LLM usage
        if not should_have_crisis and llm_used:
            print(f"✅ PASS: LLM used for non-crisis message")
        elif should_have_crisis and not llm_used:
            print(f"✅ PASS: LLM skipped for crisis message (template used instead)")
        else:
            print(f"⚠️  INFO: LLM usage: {llm_used}")

if __name__ == "__main__":
    try:
        test_bot_crisis_logic()
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
