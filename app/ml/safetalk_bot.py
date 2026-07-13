"""
SafeTalk Bot - AI-powered crisis counseling chatbot
Uses intent-based NLP with trained model for mental health support
"""

import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os
from pathlib import Path

# Define training data for the bot
TRAINING_DATA = {
    "emergency": {
        "keywords": ["suicide", "kill myself", "hurt myself", "self harm", "emergency", "crisis", "help now"],
        "responses": [
            "I hear that you're in crisis. Your life matters. Please reach out to emergency services immediately.",
            "If you're having thoughts of self-harm, please call emergency services or the suicide prevention hotline now.",
            "I'm concerned about your safety. Please contact a mental health professional or crisis hotline immediately.",
            "You deserve immediate professional help. Please reach out to local emergency services or call 988 (Suicide & Crisis Lifeline)."
        ]
    },
    "depression": {
        "keywords": ["depressed", "sad", "worthless", "hopeless", "nothing matters", "empty"],
        "responses": [
            "I'm sorry you're feeling this way. Depression is treatable. Have you considered talking to a counselor?",
            "These feelings are valid, but they can improve with proper support. Would you like to talk to someone?",
            "It's okay to feel sad sometimes, but persistent depression needs help. Professional support can make a real difference.",
            "You're not alone in feeling this way. Many people have recovered from depression with proper treatment."
        ]
    },
    "anxiety": {
        "keywords": ["anxious", "anxiety", "nervous", "worried", "panic", "stressed"],
        "responses": [
            "Anxiety is manageable. Let's try some breathing exercises: inhale for 4 counts, hold for 4, exhale for 4.",
            "What you're feeling is treatable. Have you tried relaxation techniques like meditation or deep breathing?",
            "Anxiety often gets worse when we're alone. Talking to someone can really help. Do you want to try?",
            "Your feelings are valid. Professional support can help you develop coping strategies for anxiety."
        ]
    },
    "coping": {
        "keywords": ["cope", "manage", "handle", "deal with", "strategy", "help"],
        "responses": [
            "Some healthy coping strategies include exercise, talking to friends, journaling, or mindfulness.",
            "Have you tried activities that help you relax? Exercise, hobbies, or spending time in nature can help.",
            "Professional counselors can teach you specific coping strategies tailored to your needs.",
            "Building a support network is important. Do you have people you trust to talk to?"
        ]
    },
    "support": {
        "keywords": ["need help", "support", "talk to someone", "counselor", "therapist", "advice"],
        "responses": [
            "I'm glad you're reaching out. Professional support is available. Would you like to chat with a counselor?",
            "Seeking help is a sign of strength. Our counselors are here to support you.",
            "You don't have to face this alone. Connecting with a professional counselor can really help.",
            "I'm here to listen and help connect you with the right resources."
        ]
    },
    "gratitude": {
        "keywords": ["thank you", "thanks", "appreciate", "helpful", "better", "improving"],
        "responses": [
            "I'm glad I could help. Keep taking care of yourself and reach out anytime you need support.",
            "You're doing great. Remember to be kind to yourself and celebrate small victories.",
            "That's wonderful to hear. Please continue reaching out when you need support.",
            "Your wellbeing matters. Keep building on this progress!"
        ]
    }
}

class SafeTalkBot:
    def __init__(self, model_path=None):
        self.model_path = model_path or str(Path(__file__).parent.parent.parent / "ml_models" / "safetalk_bot.pkl")
        self.vectorizer = None
        self.classifier = None
        self.intents_data = TRAINING_DATA
        self._train_model()
    
    def _train_model(self):
        """Train the bot with intent classification"""
        X_train = []
        y_train = []
        
        # Prepare training data
        for intent, data in self.intents_data.items():
            for keyword in data["keywords"]:
                X_train.append(keyword)
                y_train.append(intent)
        
        # Create pipeline
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english')),
            ('nb', MultinomialNB())
        ])
        
        # Train
        self.classifier.fit(X_train, y_train)
    
    def _detect_intent(self, message: str) -> str:
        """Detect intent from user message"""
        message_lower = message.lower()
        
        # Check for direct keyword matches first (higher priority)
        for intent, data in self.intents_data.items():
            for keyword in data["keywords"]:
                if keyword in message_lower:
                    return intent
        
        # Use classifier as fallback
        try:
            if self.classifier:
                predicted_intent = self.classifier.predict([message])[0]
                return predicted_intent
        except:
            pass
        
        return "support"  # Default intent
    
    def get_response(self, user_message: str) -> dict:
        """
        Generate a bot response based on user message
        
        Args:
            user_message: The user's input message
            
        Returns:
            dict with response and metadata
        """
        import random
        
        # Detect intent
        intent = self._detect_intent(user_message)
        
        # Get response
        responses = self.intents_data[intent]["responses"]
        response = random.choice(responses)
        
        return {
            "response": response,
            "intent": intent,
            "confidence": 0.85,
            "suggestions": self._get_suggestions(intent)
        }
    
    def _get_suggestions(self, intent: str) -> list:
        """Get follow-up suggestions based on intent"""
        suggestions = {
            "emergency": ["Call 988", "Contact local emergency", "Go to nearest hospital"],
            "depression": ["Talk to counselor", "Share with someone", "Join group session"],
            "anxiety": ["Try breathing exercise", "Chat with counselor", "Learn coping strategies"],
            "coping": ["Practice regularly", "Join support group", "Track your progress"],
            "support": ["Schedule session", "Video chat counselor", "Browse resources"],
            "gratitude": ["Continue self-care", "Set goals", "Plan next session"]
        }
        return suggestions.get(intent, ["Chat with counselor", "Schedule follow-up"])
    
    def save_model(self):
        """Save trained model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'intents': self.intents_data
            }, f)
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.classifier = data['classifier']
                self.intents_data = data['intents']

# Initialize bot instance
bot_instance = None

def get_bot():
    """Get or create bot instance"""
    global bot_instance
    if bot_instance is None:
        bot_instance = SafeTalkBot()
    return bot_instance
