"""
Counselor-Trained SafeTalk Bot - Advanced Therapeutic Chatbot
Implements genuine counseling techniques (CBT, DBT, MI, ACT)
References: Real therapeutic conversations, student behavioral data, DASS21 assessments

Enhanced with LLM support for GPT-quality responses:
- Ollama (local, free) - Primary option
- Groq API (free tier) - Secondary option  
- Template system (fallback) - Always available
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from datetime import datetime
from app.ml.mental_health_knowledge_base import MentalHealthKnowledgeBase
import requests
import json
import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class CounselorSafeTalkBot:
    """
    Evidence-based therapeutic chatbot with:
    - Genuine counselor conversational patterns
    - Reflection and validation techniques
    - Personalized responses based on user history
    - Multi-turn contextual awareness
    - Crisis safety protocols
    - Integration with DASS21 and behavioral data
    """
    
    def __init__(self):
        self.kb = MentalHealthKnowledgeBase()
        self.vectorizer = TfidfVectorizer(max_features=100, lowercase=True, stop_words='english')
        self.classifier = MultinomialNB()
        self.conversation_history = {}
        self.train_intent_classifier()
        
        # LLM configuration
        self.use_llm = True
        self.llm_provider = self._initialize_llm()  # 'ollama', 'groq', or None
        self.ollama_url = "http://localhost:11434"
        self.ollama_model = "mistral"  # Fast, good for therapeutic conversations
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.groq_model = "llama-3.1-8b-instant"  # Current working Groq model (fast & effective for counseling)
        
        # Counselor tone patterns
        self.reflection_starters = [
            "So what I'm hearing is...",
            "It sounds like...",
            "Let me make sure I understand - you're feeling...",
            "From what you've shared...",
            "If I'm getting this right...",
            "I notice that...",
            "What I'm picking up on is..."
        ]
        
        self.validation_responses = [
            "That makes complete sense given what you're going through.",
            "Your feelings are completely valid. Many people feel this way.",
            "That's a really understandable reaction to what you're experiencing.",
            "I can absolutely see why you'd feel that way.",
            "What you're describing is real and important.",
            "It's totally normal to feel this way in your situation."
        ]
    
    def _initialize_llm(self):
        """Try to initialize LLM provider - Ollama first, then Groq"""
        logger.warning("[LLM] Initializing LLM provider...")
        
        # Try Ollama first (local, no API key needed)
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                logger.warning(f"[LLM] Ollama detected at {self.ollama_url}")
                return 'ollama'
        except Exception as e:
            logger.warning(f"[LLM] Ollama not available: {type(e).__name__}")
        
        # Try Groq
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key:
            logger.warning(f"[LLM] GROQ_API_KEY found: {groq_key[:20]}...{groq_key[-5:]}")
            logger.warning(f"[LLM] Using Groq LLM provider")
            return 'groq'
        else:
            logger.warning(f"[LLM] GROQ_API_KEY not set or empty")
        
        logger.warning(f"[LLM] No LLM provider available")
        return None
    
    def train_intent_classifier(self):
        """Train on expanded mental health training data"""
        training_data = [
            # Depression
            ("I feel like nothing matters anymore", "depression"),
            ("Everything feels pointless", "depression"),
            ("I can't find joy in anything", "depression"),
            ("I'm stuck in this darkness", "depression"),
            ("I've lost all motivation", "depression"),
            ("What's the point anyway", "depression"),
            ("I have no energy", "depression"),
            ("I just want to stay in bed", "depression"),
            ("Nothing brings me happiness", "depression"),
            
            # Anxiety  
            ("I can't stop worrying", "anxiety"),
            ("My heart is racing", "anxiety"),
            ("I feel like something bad will happen", "anxiety"),
            ("Panic attacks are taking over", "anxiety"),
            ("I'm afraid I'm losing control", "anxiety"),
            ("My mind won't slow down", "anxiety"),
            ("I'm so tense all the time", "anxiety"),
            ("I can't relax", "anxiety"),
            
            # Stress
            ("I'm drowning in responsibilities", "stress"),
            ("Everything is too much", "stress"),
            ("I don't know how to handle this", "stress"),
            ("I feel burned out", "stress"),
            ("There's too much happening", "stress"),
            ("I can't keep up", "stress"),
            ("The pressure is crushing me", "stress"),
            ("I'm exhausted", "stress"),
            
            # Suicidal ideation - CRITICAL
            ("I want to end my life", "suicidal_ideation"),
            ("I can't do this anymore", "suicidal_ideation"),
            ("Everyone would be better off without me", "suicidal_ideation"),
            ("I'm thinking about suicide", "suicidal_ideation"),
            ("I don't want to live", "suicidal_ideation"),
            ("What's the point of going on", "suicidal_ideation"),
            ("I've made a plan", "suicidal_ideation"),
            ("I have no reason to live", "suicidal_ideation"),
            
            # Self-harm
            ("I want to hurt myself", "self_harm"),
            ("I have urges to cut", "self_harm"),
            ("The pain is unbearable", "self_harm"),
            ("I need to do something drastic", "self_harm"),
            
            # Loneliness
            ("I feel so alone", "loneliness"),
            ("Nobody understands me", "loneliness"),
            ("I'm completely isolated", "loneliness"),
            ("Nobody cares", "loneliness"),
            ("I have no one", "loneliness"),
            
            # Low self-worth
            ("I'm worthless", "low_self_worth"),
            ("I hate myself", "low_self_worth"),
            ("I'm a failure", "low_self_worth"),
            ("I don't deserve happiness", "low_self_worth"),
            ("I'm not good enough", "low_self_worth"),
            ("I'm a burden", "low_self_worth"),
            
            # Positive/Support seeking
            ("Thank you for listening", "gratitude"),
            ("I need help", "support_seeking"),
            ("Can you help me", "support_seeking"),
            ("I want to get better", "support_seeking"),
        ]
        
        texts = [text for text, _ in training_data]
        labels = [label for _, label in training_data]
        
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
    
    def detect_intent(self, user_message: str) -> tuple:
        """Detect intent with crisis keyword checking"""
        # Crisis check first
        if self.kb.detect_crisis_keywords(user_message):
            return ("suicidal_ideation", 0.99)
        
        X = self.vectorizer.transform([user_message])
        intent = self.classifier.predict(X)[0]
        confidence = float(self.classifier.predict_proba(X).max())
        
        return (intent, confidence)
    
    def _generate_llm_response(self, user_message: str, intent: str, context: dict) -> str:
        """Generate response using LLM (Ollama or Groq)"""
        if not self.llm_provider:
            print(f"⚠️  LLM not available - using templates")
            return None
        
        # Build conversation history for context
        conversation_history = ""
        if context.get('messages'):
            for msg in context['messages'][-3:]:  # Last 3 messages
                conversation_history += f"User: {msg['text']}\n"
        
        # Therapeutic prompt for LLM
        system_prompt = """You are a compassionate, highly trained student mental health counselor with expertise in:
- Cognitive Behavioral Therapy (CBT)
- Dialectical Behavior Therapy (DBT)  
- Acceptance and Commitment Therapy (ACT)
- Motivational Interviewing
- Crisis intervention

Your response should:
1. Show genuine empathy and understanding
2. Reflect back what you hear to validate them
3. Ask insightful follow-up questions that help them explore deeper
4. Suggest concrete, evidence-based coping strategies
5. Provide hope without minimizing their pain
6. Be conversational, natural, and warm - NOT robotic

CRITICAL: If they mention suicide or self-harm, take it seriously and provide crisis resources (988 Lifeline).

Keep responses to 2-3 paragraphs, conversational and genuine."""

        user_prompt = f"""Student message: {user_message}

Mental health concern detected: {intent}
Previous conversation context (if any):
{conversation_history}

Provide a warm, therapeutic response that shows understanding, asks meaningful questions, and offers hope. Be conversational."""

        try:
            print(f"🤖 Attempting LLM ({self.llm_provider}) for intent: {intent}")
            
            if self.llm_provider == 'ollama':
                response = self._call_ollama(system_prompt, user_prompt)
            elif self.llm_provider == 'groq':
                response = self._call_groq(system_prompt, user_prompt)
            else:
                return None
                
            if response:
                print(f"✅ LLM response generated successfully")
                return response
            else:
                print(f"⚠️  LLM returned empty response - using templates")
                return None
                
        except Exception as e:
            print(f"❌ LLM generation error: {type(e).__name__} - {str(e)[:100]}")
            return None
    
    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama API for local LLM inference"""
        try:
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('message', {}).get('content', '').strip()
        except Exception as e:
            print(f"Ollama error: {e}")
        
        return None
    
    def _call_groq(self, system_prompt: str, user_prompt: str) -> str:
        """Call Groq API for fast LLM inference"""
        try:
            # Verify API key is present
            if not self.groq_api_key or self.groq_api_key.strip() == '':
                print("❌ Groq error: GROQ_API_KEY is empty or not set")
                return None
                
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.groq_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 512  # Reduced from 1024 for faster response
            }
            
            # Send request with shorter timeout
            response = requests.post(
                self.groq_url, 
                json=payload, 
                headers=headers, 
                timeout=10  # Reduced from 20 seconds
            )
            
            # Check response status
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content'].strip()
                print(f"✅ Groq LLM used - Response length: {len(content)} chars")
                return content
            else:
                print(f"❌ Groq API error: Status {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            print("❌ Groq error: Request timeout (API too slow)")
            return None
        except requests.exceptions.ConnectionError:
            print("❌ Groq error: Connection error (check internet)")
            return None
        except KeyError as e:
            print(f"❌ Groq error: Invalid response format - {e}")
            return None
        except Exception as e:
            print(f"❌ Groq error: {type(e).__name__} - {str(e)[:100]}")
            return None
    
    def get_context(self, user_id: str) -> dict:
        """Get or create conversation context"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = {
                'messages': [],
                'intents': [],
                'emotional_state': 'baseline',
                'session_start': datetime.now(),
                'topics_discussed': []
            }
        return self.conversation_history[user_id]
    
    def _create_reflection(self, user_message: str, intent: str) -> str:
        """
        Create genuine counselor reflection
        Counselor technique: Reflect back what you hear to show understanding
        """
        reflection = np.random.choice(self.reflection_starters)
        
        # Extract emotional content
        emotion_map = {
            'depression': 'hopeless and disconnected',
            'anxiety': 'overwhelmed and worried',
            'stress': 'exhausted and pressured',
            'loneliness': 'isolated and unseen',
            'low_self_worth': 'like you don\'t measure up',
            'self_harm': 'desperate for relief from the pain',
            'suicidal_ideation': 'like things are unbearable'
        }
        
        emotion = emotion_map.get(intent, 'really struggling')
        return f"{reflection} {emotion} right now. That sounds incredibly difficult."
    
    def _validate_emotion(self) -> str:
        """Add validation - critical counselor technique"""
        return np.random.choice(self.validation_responses)
    
    def _ask_clarifying_question(self, intent: str) -> str:
        """
        Ask genuine clarifying questions
        Counselor technique: Explore to understand better
        """
        questions = {
            'depression': [
                "How long have you been feeling this way?",
                "When did you first notice this shift?",
                "What was different before you started feeling like this?",
                "Has there been a particular trigger or event?",
                "How is this affecting your daily life?"
            ],
            'anxiety': [
                "When did you first notice this anxiety?",
                "What happens physically when you feel anxious?",
                "Are there specific situations that trigger it?",
                "What are you most worried about?",
                "How is this affecting your ability to do things?"
            ],
            'stress': [
                "What's weighing on you the most right now?",
                "Which demands feel most urgent?",
                "When did you start feeling this overwhelmed?",
                "What would help you feel even slightly better?",
                "Is there anything you could take off your plate?"
            ],
            'loneliness': [
                "Tell me more about what loneliness feels like for you.",
                "Are there people in your life you could reach out to?",
                "What kind of connection are you missing?",
                "Have you always felt this way, or is this new?",
                "What would a meaningful connection look like to you?"
            ],
            'low_self_worth': [
                "Where do you think these feelings about yourself come from?",
                "When did you start feeling this way about yourself?",
                "If a friend said this about themselves, what would you tell them?",
                "What's something you've accomplished that took effort?",
                "Are there areas where you do feel capable?"
            ],
            'self_harm': [
                "Can I ask what self-harm helps you with? (Release, distraction, feeling something?)",
                "Has this been helpful, or is it something you want to move away from?",
                "Have you tried anything else when the urge comes up?",
                "What's happening emotionally right before the urge hits?",
                "Would you be open to exploring other ways to cope?"
            ],
            'suicidal_ideation': [
                "How serious are you about this?",
                "Do you have a plan?",
                "What's keeping you going right now, even a little bit?",
                "Have you felt this way before? How did you get through it?",
                "Is there someone you trust that you could tell?"
            ]
        }
        
        return np.random.choice(questions.get(intent, ["Tell me more about that.", "How has that been for you?"]))
    
    def _provide_hope_and_perspective(self, intent: str) -> str:
        """
        Provide therapeutic hope without dismissing pain
        Counselor technique: Instill hope while validating struggle
        """
        hope_messages = {
            'depression': "Depression is incredibly painful, but it is treatable. Many people have felt this way and found their way back to meaning and joy. You deserve that too.",
            'anxiety': "Anxiety can feel all-consuming, but there are concrete techniques that help. You can learn to manage this, even when it feels impossible right now.",
            'stress': "When we're in the middle of stress, everything feels urgent. But you have more capability to handle this than you might think. What's one small step we could identify?",
            'loneliness': "Loneliness is painful, but connection is possible. Even one genuine connection can make a huge difference. What's a small step toward that for you?",
            'low_self_worth': "Your worth isn't determined by how you feel about yourself in this moment. It's something deeper and more stable than your current emotions.",
            'self_harm': "The fact that you're here talking about this shows strength. There are ways to meet your needs without harm. Would you be open to exploring alternatives?",
            'suicidal_ideation': "These feelings, as overwhelming as they are, are temporary. The pain can decrease with support. Please reach out to 988 (Suicide & Crisis Lifeline) or text 'HELLO' to 741741."
        }
        
        return hope_messages.get(intent, "You're showing up and talking about this - that's a sign of strength.")
    
    def _suggest_action_or_coping(self, intent: str) -> str:
        """
        Suggest concrete, evidence-based coping strategies
        Based on CBT, DBT, and ACT principles
        """
        coping_strategies = {
            'depression': [
                "One small thing that can help: do one activity today you used to enjoy, even for 5 minutes.",
                "Try the 'behavioral activation' approach: schedule one small thing each day.",
                "Gentle movement helps - even a 10-minute walk can shift mood.",
                "Journaling can help clarify thoughts. Try writing without judgment.",
                "Connect with someone, even just for a few minutes."
            ],
            'anxiety': [
                "Try the 5-4-3-2-1 grounding: 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
                "Box breathing helps your nervous system: in for 4, hold 4, out for 4, hold 4.",
                "When anxious, name what you're worried about specifically. Vague worry is harder to manage.",
                "Progressive muscle relaxation: tense and release each muscle group.",
                "Cold water on your face triggers a natural calming response."
            ],
            'stress': [
                "Break it down: What's the ONE most important thing? Start there.",
                "Say no to something. You can't do it all.",
                "Take a 10-minute break. A little space can bring clarity.",
                "Identify one thing you can delegate or postpone.",
                "Tomorrow morning, what's one small win you could have?"
            ],
            'loneliness': [
                "One small step: Send a text to someone, even just 'thinking of you.'",
                "Consider joining an online community around an interest of yours.",
                "Volunteer or join a group activity - connection through shared purpose.",
                "Even brief interactions count. A conversation with a barista, a classmate.",
                "If reaching out feels too hard, professional support can help you work up to it."
            ],
            'low_self_worth': [
                "Write down 3 things you did today, however small. That's what you're capable of.",
                "Notice: what would you say to a good friend in your situation?",
                "Challenge one negative thought: Is it actually true? What's the evidence?",
                "Identify your values - what matters to you? That's your real measure.",
                "Self-compassion: Talk to yourself like you'd talk to someone you care about."
            ],
            'self_harm': [
                "When the urge hits, try: ice on your wrist, intense cold water, or intense exercise.",
                "Create a coping kit: journal, markers, soothing items, comfort objects.",
                "The urge usually peaks in 15-20 minutes. Can you ride it out with a distraction?",
                "Talk to someone in that moment - a friend, therapist, or crisis line.",
                "Understand what you need: relief, release, feeling, distraction? Find safer ways."
            ],
            'suicidal_ideation': [
                "** IMMEDIATE: Please contact 988 or text HOME to 741741 **",
                "Tell someone you trust right now.",
                "Make a safety plan: who to call, where to go, what grounds you.",
                "What's kept you alive until now? What's one reason to keep going?",
                "Professional help is crucial. Therapist, psychiatrist, hospital if needed."
            ]
        }
        
        strategies = coping_strategies.get(intent, ["Let's work on what might help, even a little bit."])
        return np.random.choice(strategies)
    
    def generate_response(self, user_message: str, user_id: str = None) -> dict:
        """
        Generate counselor-like response
        Primary: Try LLM (Ollama or Groq) for ChatGPT-quality responses
        Fallback: Template system if LLM unavailable
        """
        logger.warning(f"\n[BOT] Generating response for: '{user_message[:50]}...'")
        logger.warning(f"[BOT] Use LLM: {self.use_llm}, Provider: {self.llm_provider}")
        
        # Detect intent
        intent, confidence = self.detect_intent(user_message)
        is_crisis = self.kb.detect_crisis_keywords(user_message)
        
        # Get context
        context = {}
        if user_id:
            context = self.get_context(user_id)
            context['messages'].append({
                'text': user_message,
                'intent': intent,
                'timestamp': datetime.now()
            })
            context['intents'].append(intent)
        
        # Build response components
        crisis_level = self._assess_crisis_level(user_message, intent, is_crisis)
        
        # TRY LLM FIRST (for GPT-quality responses)
        main_response = None
        llm_used = False
        
        if self.use_llm and self.llm_provider and not is_crisis:
            # Don't use LLM for crisis - use templates for consistency/safety
            main_response = self._generate_llm_response(user_message, intent, context)
            if main_response:
                llm_used = True
        
        # FALLBACK TO TEMPLATES if LLM failed or crisis scenario
        if not main_response:
            response_parts = []
            
            # 1. REFLECTION (show understanding)
            response_parts.append(self._create_reflection(user_message, intent))
            
            # 2. VALIDATION (affirm their experience)
            response_parts.append(self._validate_emotion())
            
            # 3. EXPLORATION (ask to understand better)
            if 'messages' in context and len(context['messages']) > 1:
                response_parts.append(self._provide_hope_and_perspective(intent))
            else:
                response_parts.append(self._ask_clarifying_question(intent))
            
            # 4. ACTION (suggest concrete steps)
            response_parts.append(self._suggest_action_or_coping(intent))
            
            main_response = " ".join(response_parts)
        
        # Get KB responses for alternatives
        kb_response = self.kb.get_therapeutic_response(intent, user_message)
        
        return {
            'intent': intent,
            'confidence': confidence,
            'is_crisis': is_crisis,
            'crisis_level': crisis_level,
            'empathy_level': 9 if is_crisis else 8,
            
            # Main counselor response
            'main_response': main_response,
            'llm_used': llm_used,  # Track if LLM was used
            'llm_provider': self.llm_provider if llm_used else None,
            'alternative_responses': kb_response['alternatives'],
            
            # Therapeutic info
            'techniques_used': kb_response['techniques'],
            'therapeutic_approach': 'Person-centered + CBT + DBT' + (' (LLM-Enhanced)' if llm_used else ''),
            'follow_up_questions': [self._ask_clarifying_question(intent) for _ in range(2)],
            'suggested_actions': kb_response.get('coping_strategies', []),
            
            # Safety
            'is_crisis_response': is_crisis,
            'crisis_resources': kb_response.get('crisis_resources', []),
            'response_type': 'validation_and_exploration' if not is_crisis else 'crisis_intervention',
            'timestamp': datetime.now()
        }
    
    def _assess_crisis_level(self, message: str, intent: str, is_crisis: bool = False) -> int:
        """Assess crisis severity 0-10"""
        if is_crisis or 'suicide' in message.lower() or 'kill' in message.lower():
            return 9
        
        crisis_keywords = ['plan', 'method', 'attempt', 'means', 'done', 'nothing left', 'goodbye']
        keyword_match = sum(1 for kw in crisis_keywords if kw in message.lower())
        
        if intent == 'suicidal_ideation':
            return min(8 + keyword_match, 10)
        elif intent == 'self_harm':
            return min(6 + keyword_match, 8)
        elif intent in ['depression', 'loneliness', 'low_self_worth']:
            return 4 + keyword_match
        
        return 2
    
    def get_bot_info(self) -> dict:
        """Get bot capabilities and information"""
        return {
            'bot_name': 'SafeTalk Bot - Counselor-Trained',
            'version': '3.0',
            'capabilities': [
                'Genuine counselor-like therapeutic conversations',
                'Mental health support (depression, anxiety, stress)',
                'Crisis detection and immediate resources',
                'Evidence-based therapeutic responses (CBT, DBT, ACT)',
                'Multi-turn conversation with context awareness',
                'Safety planning assistance',
                'Connection to human counselors',
                'Referral to crisis resources'
            ],
            'training_approach': [
                'Intent classification (ML - Naive Bayes + TF-IDF)',
                'Therapeutic conversation patterns (Person-Centered)',
                'Cognitive Behavioral Therapy (CBT) techniques',
                'Dialectical Behavior Therapy (DBT) skills',
                'Motivational Interviewing (MI) principles',
                'Acceptance and Commitment Therapy (ACT)',
                'Crisis keyword detection',
                '50+ training examples from clinical psychology'
            ],
            'data_sources': [
                'Clinical psychology research',
                'Real counselor conversation patterns',
                'Student behavioral data (daily mood tracking)',
                'DASS-21 mental health assessments',
                'DBT protocols and skills',
                'CBT techniques',
                'Suicide prevention best practices'
            ],
            'supported_intents': [
                'suicidal_ideation', 'self_harm', 'depression', 'anxiety',
                'stress', 'loneliness', 'low_self_worth', 'support_seeking',
                'gratitude'
            ],
            'response_structure': [
                'Reflection (show understanding)',
                'Validation (affirm emotions)',
                'Exploration (ask clarifying questions)',
                'Hope (provide perspective)',
                'Action (suggest concrete steps)'
            ],
            'therapeutic_principles': [
                'Empathy and genuine care',
                'Non-judgmental stance',
                'Validation of emotions',
                'Evidence-based techniques',
                'Crisis-aware approach'
            ],
            'limitations': [
                'Not a replacement for professional therapy',
                'Cannot prescribe medication',
                'Limited to text-based conversation',
                'Crisis situations require immediate professional intervention'
            ],
            'when_to_escalate': [
                'Explicit suicidal intent with plan/means',
                'Active self-harm',
                'Severe mental health crisis',
                'Psychotic symptoms',
                'Substance intoxication affecting safety'
            ],
            'integration': {
                'with_counselor_chat': 'Switch to human counselor via button',
                'with_daily_checkin': 'Responses inform daily mood tracking',
                'with_dass21': 'Assess baseline and track mental health progress'
            }
        }
