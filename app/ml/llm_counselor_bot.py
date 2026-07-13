"""
LLM-Powered Counselor Bot using Ollama/Local Models
Provides more natural, contextually-aware therapeutic responses
Uses Mistral, Llama 2, or other open-source models
"""

import os
import re
import time
from datetime import datetime
from typing import Dict, Tuple, Optional
from app.ml.mental_health_knowledge_base import MentalHealthKnowledgeBase

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama not installed. Install from: https://ollama.ai")


class LLMCounselorBot:
    """
    LLM-based counselor bot with local model support
    Falls back to rule-based system if Ollama unavailable
    """
    
    def __init__(self, model_name: str = "mistral"):
        """
        Initialize LLM counselor bot
        
        Args:
            model_name: Ollama model ('mistral', 'llama2', 'neural-chat', etc.)
        """
        self.kb = MentalHealthKnowledgeBase()
        self.model_name = model_name
        self.conversation_history = {}
        self.response_history = {}  # Track used responses per user
        self.model_available = OLLAMA_AVAILABLE
        self.model_ready = False
        
        if self.model_available:
            try:
                # Test connection and model availability
                self._ensure_model_loaded()
                self.model_ready = True
            except Exception as e:
                print(f"Warning: LLM model initialization failed: {e}")
                self.model_ready = False
        
        # System prompt for counselor behavior
        self.system_prompt = """You are SafeTalk Bot, a compassionate mental health counselor trained in evidence-based therapeutic techniques including:
- Cognitive Behavioral Therapy (CBT)
- Dialectical Behavior Therapy (DBT)
- Motivational Interviewing (MI)
- Person-Centered Counseling
- Acceptance & Commitment Therapy (ACT)

Your response should:
1. REFLECT: Show you understand their feelings
2. VALIDATE: Affirm their emotions are real and understandable
3. EXPLORE: Ask deeper questions to understand context
4. HOPE: Provide perspective and support
5. ACTION: Suggest practical coping strategies

Important guidelines:
- Be empathetic and warm, never clinical
- Use active listening techniques
- Ask one or two clarifying questions
- Suggest evidence-based coping strategies
- Always prioritize safety - if crisis suspected, recommend professional help
- Keep responses conversational and natural (1-2 paragraphs max)
- Never pretend to be a substitute for professional mental health services

Conversation tone: Warm, understanding, professional but approachable. Like talking to a trusted counselor."""

        # Crisis keywords to monitor
        self.crisis_keywords = {
            10: [
                "i want to die", "i'm going to kill myself", "i'm killing myself tonight",
                "suicide", "end it all", "no point living", "better off dead",
                "rope", "pills", "overdose", "jump from", "gun", "take my life"
            ],
            9: [
                "suicidal", "suicide ideation", "i won't be around", "won't make it",
                "see tomorrow", "last goodbye", "final message", "not going to live"
            ],
            7: [
                "want to harm myself", "cutting", "self harm", "depressed beyond",
                "hopeless situation", "can't take it anymore", "want to end this"
            ],
            5: [
                "really depressed", "anxious all the time", "having panic attacks",
                "can't cope", "overwhelmed", "stressed out", "falling apart"
            ]
        }
    
    def _ensure_model_loaded(self):
        """Ensure Ollama model is available locally"""
        if not self.model_available:
            return
        
        try:
            # Try to get model info
            models = ollama.list()
            model_names = [m['name'].split(':')[0] for m in models['models']]
            
            if self.model_name not in model_names:
                print(f"\n📥 Downloading {self.model_name} model (this may take a few minutes)...")
                ollama.pull(self.model_name)
        except Exception as e:
            print(f"Model loading error: {e}")
            raise
    
    def _assess_crisis_level(self, text: str) -> int:
        """
        Assess crisis level from 0-10 scale
        
        Returns:
            int: Crisis level (0=safe, 10=immediate danger)
        """
        text_lower = text.lower()
        
        # Check highest severity first
        for level in sorted(self.crisis_keywords.keys(), reverse=True):
            for keyword in self.crisis_keywords[level]:
                if keyword in text_lower:
                    return level
        
        return 0
    
    def _build_context(self, user_id: int, user_message: str) -> str:
        """
        Build rich conversation context for LLM
        
        Args:
            user_id: User identifier
            user_message: Current user message
            
        Returns:
            str: Formatted context string
        """
        # Get conversation history (last 5 messages)
        history_context = ""
        if user_id in self.conversation_history:
            recent = self.conversation_history[user_id][-5:]
            for msg in recent:
                history_context += f"User: {msg['user']}\nCounselor: {msg['bot']}\n"
        
        # Build context string
        context = f"""Previous conversation:
{history_context if history_context else "This is the start of the conversation."}

Current message from user:
"{user_message}"

Please respond as SafeTalk Bot counselor. Use therapeutic techniques to provide compassionate support."""
        
        return context
    
    def generate_response(self, user_message: str, user_id: int = None) -> Dict:
        """
        Generate therapeutic response using LLM or fallback
        
        Args:
            user_message: User's text
            user_id: User identifier for history
            
        Returns:
            Dict with response, crisis_level, intent, etc.
        """
        # Assess crisis level first
        crisis_level = self._assess_crisis_level(user_message)
        
        # Initialize history
        if user_id and user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        # Try LLM first, fallback to rule-based
        if self.model_ready:
            response_text = self._generate_llm_response(user_message, user_id)
        else:
            response_text = self._generate_fallback_response(user_message, crisis_level, user_id)
        
        # Add crisis specific guidance
        if crisis_level >= 9:
            response_text += "\n\n🆘 **Important**: If you're having thoughts of suicide, please reach out to a crisis service immediately:\n- National Suicide Prevention Lifeline: 988 (US)\n- Crisis Text Line: Text HOME to 741741\n- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/"
        
        # Save to history
        if user_id:
            self.conversation_history[user_id].append({
                'user': user_message,
                'bot': response_text,
                'timestamp': datetime.now()
            })
        
        return {
            'response': response_text,
            'crisis_level': crisis_level,
            'model_used': 'LLM' if self.model_ready else 'Fallback',
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.95 if self.model_ready else 0.75
        }
    
    def _generate_llm_response(self, user_message: str, user_id: Optional[int] = None) -> str:
        """
        Generate response using Ollama LLM
        
        Args:
            user_message: User's message
            user_id: User ID for context
            
        Returns:
            str: LLM-generated response
        """
        try:
            # Build context
            context = self._build_context(user_id, user_message)
            
            # Call LLM
            response = ollama.generate(
                model=self.model_name,
                prompt=context,
                system=self.system_prompt,
                stream=False,
                options={
                    'temperature': 0.7,  # Balance creativity and consistency
                    'top_p': 0.9,
                    'num_predict': 256,  # Limit response length
                }
            )
            
            # Extract response text
            bot_response = response['response'].strip()
            
            # Ensure response is therapeutic and safe
            bot_response = self._sanitize_response(bot_response)
            
            return bot_response
            
        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._generate_fallback_response(user_message, 0)
    
    def _sanitize_response(self, response: str) -> str:
        """
        Ensure response is appropriate and therapeutic
        
        Args:
            response: Raw LLM response
            
        Returns:
            str: Sanitized response
        """
        # Remove any meta text or instructions that leaked
        if "As a counselor" in response or "As SafeTalk" in response:
            lines = response.split('\n')
            response = '\n'.join(line for line in lines if not line.startswith('As'))
        
        # Ensure it starts with therapeutic element
        response = response.strip()
        if not response:
            response = "I hear you. That sounds really challenging. Can you tell me more about what's going on?"
        
        # Limit length (excessive responses are less therapeutic)
        if len(response) > 1000:
            response = response[:1000].rsplit(' ', 1)[0] + "..."
        
        return response
    
    def _select_varied_response(self, responses: list, user_id: int = None, category: str = "default") -> str:
        """
        Select response while avoiding repetition for same user
        
        Args:
            responses: List of possible responses
            user_id: User ID for tracking
            category: Response category for tracking
            
        Returns:
            str: Selected response
        """
        import random
        
        if not responses:
            return "I'm here to listen. Tell me what's going on."
        
        if user_id:
            if user_id not in self.response_history:
                self.response_history[user_id] = {}
            
            if category not in self.response_history[user_id]:
                self.response_history[user_id][category] = set()
            
            # Get responses not recently used
            unused = [r for r in responses if r not in self.response_history[user_id][category]]
            
            # If all used, reset and pick new ones
            if not unused:
                self.response_history[user_id][category] = set()
                unused = responses
            
            selected = random.choice(unused)
            self.response_history[user_id][category].add(selected)
            
            # Keep only last 3 to allow some recycling
            if len(self.response_history[user_id][category]) > 3:
                oldest = list(self.response_history[user_id][category])[0]
                self.response_history[user_id][category].discard(oldest)
            
            return selected
        else:
            return random.choice(responses)
    
    def _generate_fallback_response(self, user_message: str, crisis_level: int, user_id: int = None) -> str:
        """
        Enhanced fallback response with natural, varied, relatable language
        
        Args:
            user_message: User message
            crisis_level: Crisis assessment level
            user_id: User ID for tracking response variety
            
        Returns:
            str: Therapeutic response
        """
        message_lower = user_message.lower()
        
        # Crisis responses - varied
        if crisis_level >= 9:
            crisis_responses = [
                "I'm really concerned about what you're sharing right now. This sounds urgent and serious. Please reach out to a crisis service immediately - they have trained professionals who specialize in exactly this kind of situation:\n\n🆘 988 (Suicide & Crisis Lifeline)\n💬 Text HOME to 741741\n\nYou deserve immediate support, and these people are genuinely trained to help in moments like this.",
                "What you're describing is a crisis, and I want you to know something important: **you don't have to handle this alone**. Please contact emergency services or a crisis line right now. There are people trained for exactly this moment, and they can help:\n\n• National Suicide Prevention Lifeline: 988\n• Crisis Text Line: Text HOME to 741741\n• Emergency: Call 911 or go to nearest ER\n\nYou matter, and help is available right now.",
            ]
            return self._select_varied_response(crisis_responses, user_id, "crisis_9")
        
        if crisis_level >= 7:
            high_risk_responses = [
                "I hear how serious this is, and I'm genuinely concerned about your wellbeing. What you're describing suggests you need support from someone trained in crisis situations. Would you be open to speaking with a crisis counselor or going to an emergency room? They can provide the kind of immediate, specialized help that could really make a difference.",
                "This sounds really intense, and I want to be honest - it sounds like you need more than I can provide in this chat. Have you considered reaching out to a therapist, calling a crisis line, or going to the ER? These trained professionals can offer real, immediate support."
            ]
            return self._select_varied_response(high_risk_responses, user_id, "crisis_7")
        
        # Anxiety/worry - varied responses
        if any(kw in message_lower for kw in ['anxious', 'anxiety', 'panic', 'worry', 'nervous']):
            anxiety_responses = [
                "That anxious feeling you're experiencing is real, and a lot of people relate to it. When anxiety shows up, it often feels like your mind is running a worst-case scenario on repeat. One thing that helps some people is grounding techniques - basically bringing your attention back to what's happening *right now* instead of what-ifs. Want to try one? Notice 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
                "Anxiety has a way of making everything feel urgent and threatening, even when we're physically safe. What you're feeling is your brain trying to protect you - it just sometimes goes into overdrive. Have you noticed any patterns about when this anxiety hits hardest? Sometimes understanding the trigger helps us respond differently.",
                "I get it - that racing mind, the feeling that something bad is about to happen. Anxiety is exhausting. Here's something that might help: anxiety actually needs our fear to survive. When we can observe it without judgment (like 'okay, there's that anxious thought again') it often loses some of its power. What do you think is at the root of what you're worried about?",
            ]
            return self._select_varied_response(anxiety_responses, user_id, "anxiety")
        
        # Depression/hopelessness - varied responses
        if any(kw in message_lower for kw in ['sad', 'depressed', 'depression', 'hopeless', 'worthless', 'empty']):
            depression_responses = [
                "Depression has this way of making everything look gray and pointless - like nothing matters and nothing will get better. But here's what I know: that's the depression talking, not the truth. You're reaching out right now, which means part of you is looking for something different. What's one small thing - and I mean *small* - that's felt okay lately? Even tiny things count.",
                "What you're describing sounds like real pain. Depression is insidious because it convinces us that things won't change, that we're the problem. But depression is a liar. It distorts how we see everything. Have you ever had a moment - even briefly - where things felt slightly different? What changed in that moment?",
                "That heaviness and hopelessness you're feeling is so real, and I'm not going to dismiss it. But I also want to gently push back on one thing: depression is temporary, even though it doesn't feel that way right now. Your brain chemistry is involved here, not just your circumstances. Have you talked to anyone about how long you've been feeling this way?",
                "I hear hopelessness in what you're saying, and that's a real signal that you need support. When depression gets this heavy, sometimes we need more than just talking - therapy, maybe medication, professional support. What would it take for you to reach out to a counselor or doctor about this?",
            ]
            return self._select_varied_response(depression_responses, user_id, "depression")
        
        # Loneliness/isolation - varied responses
        if any(kw in message_lower for kw in ['alone', 'lonely', 'isolated', 'friendless', 'nobody understands']):
            loneliness_responses = [
                "Loneliness is one of the toughest feelings because it makes you feel invisible. But the fact that you're reaching out to me right now? That means you're looking for connection, and that matters. Is there one person - even someone you haven't talked to in a while - you could reach out to? Sometimes just a small connection can shift things.",
                "That isolation you're experiencing is real, but here's the thing - you're not actually alone in feeling alone. Lots of people experience this, especially when they're struggling. What would help you feel less isolated? Is it having someone to talk to, joining a group with shared interests, or something else?",
                "Being lonely can make us withdraw even more, which ironically makes the loneliness deeper. It's like a trap. Breaking that cycle sometimes means doing something that feels uncomfortable - like reaching out, joining a club, or messaging someone. What would feel manageable as a first step?",
            ]
            return self._select_varied_response(loneliness_responses, user_id, "loneliness")
        
        # Stress/overwhelm - varied responses
        if any(kw in message_lower for kw in ['stress', 'overwhelmed', 'can\'t handle', 'stressed out', 'too much']):
            stress_responses = [
                "When you're overwhelmed like this, your brain is basically short-circuiting because there's too much input. The good news? This is manageable. Instead of looking at the whole mountain, what if we just look at today - or even the next hour? What's the one thing that feels most urgent?",
                "Feeling overwhelmed often means you're carrying something that needs to be put down, delegated, or broken into smaller chunks. What's taking up the most space in your mind right now? Let's focus there.",
                "That feeling of everything piling up on you - I know it well, and it's exhausting. Here's a reality check though: you can't do everything at once, and you weren't meant to. What matters most to you right now? Start there instead of everywhere.",
                "Overwhelm usually tells us that something needs to change - we can't keep doing what we're doing the same way. What do you think is most unsustainable about your situation right now? What would help most?",
            ]
            return self._select_varied_response(stress_responses, user_id, "stress")
        
        # Relationship/conflict - varied responses
        if any(kw in message_lower for kw in ['relationship', 'boyfriend', 'girlfriend', 'spouse', 'family', 'friend', 'conflict', 'argue']):
            relationship_responses = [
                "Relationship struggles are some of the most painful because they mix hurt with rejection. What you're experiencing matters. Do you want to talk through what happened, or do you need to figure out what you actually want from this relationship?",
                "When relationships are struggling, it often comes down to unmet expectations or miscommunication. Sometimes it's fixable, sometimes it means the relationship isn't working for you. What feels true in this situation?",
                "People conflicts can really mess with your sense of belonging and self-worth. The thing is - how others treat you is about them, not about your value. That said, it's okay to feel hurt. What would help you process this?",
            ]
            return self._select_varied_response(relationship_responses, user_id, "relationships")
        
        # Work/performance - varied responses
        if any(kw in message_lower for kw in ['work', 'school', 'fail', 'grade', 'performance', 'job', 'career']):
            work_responses = [
                "Performance pressure is real, and our brains can get pretty unforgiving when we're struggling at work or school. Here's something though: one grade, one project, one day doesn't define your capability or worth. What's really going on beneath the performance anxiety?",
                "When we're struggling with work or school stuff, it can feel like we're not good enough. But usually it's more complex - it might be burnout, wrong fit, lack of support, or something else entirely. What do you think is actually going on?",
                "The pressure we put on ourselves around performance can be brutal. But remember: your productivity doesn't equal your worth. You're allowed to struggle, to need help, to not be perfect. What do you actually need right now?",
            ]
            return self._select_varied_response(work_responses, user_id, "work")
        
        # Sleep/insomnia - varied responses
        if any(kw in message_lower for kw in ['sleep', 'insomnia', 'tired', 'exhausted', 'can\'t sleep', 'sleeping', 'rest', 'nighttime']):
            sleep_responses = [
                "Sleep deprivation can really mess with everything - your mood, your thinking, your ability to cope. The fact that you're struggling with sleep tells me your body and mind need more support right now. Have you noticed if anything specific disrupts your sleep? Sometimes identifying the trigger helps us find solutions.",
                "That exhaustion you're feeling is real, and it makes everything harder. When we're sleep-deprived, our emotional resilience goes down, our stress hormones go up. It's like running a marathon on fumes. What's keeping you from getting the sleep you need? Is it your mind racing, physical discomfort, or something else?",
                "Sleep is one of those foundational things that affects everything else - mood, energy, ability to handle stress. If you're struggling with it, that matters. Some gentle approaches: stick to a schedule, avoid screens an hour before bed, try some deep breathing. But if it's persistent, a doctor can help too. What's your situation like?",
                "Tiredness can feel like a weight on everything you do. Many people experience this, especially when stressed or dealing with anxiety or depression. Sometimes it helps to establish a wind-down routine - something calming before bed. But also, if this is ongoing, it might be worth talking to someone professional about. How long has this been going on?",
            ]
            return self._select_varied_response(sleep_responses, user_id, "sleep")
        
        # Academic/School/Exam problems - varied responses
        if any(kw in message_lower for kw in ['exam', 'test', 'assignment', 'homework', 'grade', 'study', 'exam stress', 'failing', 'academic', 'gpa', 'college', 'university', 'course', 'subject', 'passing']):
            academic_responses = [
                "Academic pressure is real, and I hear it in what you're saying. Here's something important: your grades don't define your worth or your future. They're one part of a bigger picture. What's the core issue - is it understanding the material, managing time, test anxiety, or something else? Once we know, we can actually address it.",
                "School can feel overwhelming, especially when you're worried about grades or exams. That pressure you're feeling? It's normal, but it's also manageable. Let's break this down: What specific subject or assignment is stressing you most? And what have you tried so far that hasn't been working?",
                "I know exam stress can make you feel like everything depends on this one test. But here's the reality: one exam doesn't define your academic career, and you're capable of learning even if it doesn't go perfectly. What's making this particular test/assignment so stressful? Is it the content, time management, or anxiety about the results?",
                "Academic struggles are something so many students go through - you're absolutely not alone. Whether it's understanding material, managing workload, or dealing with test anxiety, there are real strategies that help. What would actually help you most right now - study strategies, time management, or help managing the anxiety itself?",
                "The stress of having to perform academically while managing everything else in life? That's a lot. And sometimes the pressure we put on ourselves makes it harder. What if we focused on what's actually within your control - your effort, your study methods, asking for help - rather than just the final grade?",
            ]
            return self._select_varied_response(academic_responses, user_id, "academic")
        
        # Relationship/Dating/Romantic issues - varied responses
        if any(kw in message_lower for kw in ['crush', 'boyfriend', 'girlfriend', 'dating', 'romance', 'romantic', 'breakup', 'ex', 'heartbreak', 'liked someone', 'love', 'intimacy', 'dating stress']):
            romance_responses = [
                "Relationship stuff as a teenager can feel so intense because everything feels heightened at this age - and that's completely valid. Whether it's a crush, a relationship stress, or a breakup, what you're feeling matters. Do you want to talk about what's happening? Sometimes just getting it out helps figure out what you actually want.",
                "Romantic feelings and relationship challenges are such a normal part of being a teenager, even though they can feel absolutely overwhelming. The thing is - how someone else treats you or feels about you? That's about them, not about your value. What's going on in your situation right now?",
                "First relationships, crushes, breakups - they're all part of growing up, but that doesn't make them hurt less. Your feelings are real even if they're new. What would help you most right now - talking through what happened, processing the emotions, or figuring out what you want to do?",
                "Navigating attraction, relationships, and maybe heartbreak while you're still figuring out who you are as a person - that's genuinely complex. There's no 'right way' to feel. What you're experiencing is valid. What do you need right now - advice, just to be heard, or help making sense of things?",
                "Relationship drama can really shake your confidence and peace of mind. But remember: your worth isn't determined by whether someone likes you back or wants to stay with you. It's about knowing yourself. What's happening that's brought this up for you?",
            ]
            return self._select_varied_response(romance_responses, user_id, "romance")
        
        # Family problems/parenting conflicts - varied responses
        if any(kw in message_lower for kw in ['family', 'parent', 'parents', 'mom', 'dad', 'brother', 'sister', 'sibling', 'home', 'household', 'family conflict', 'arguing with', 'don\'t understand', 'pressure from family', 'expectation']):
            family_responses = [
                "Family dynamics can be incredibly complicated, especially as a teenager when you're trying to figure out who you are separate from your family's expectations. What you're feeling about your family situation is valid. Are your parents/family not understanding you? Are there specific conflicts? Sometimes talking it through helps find a path forward.",
                "The thing about family is - it's so personal and multi-layered. You can love your family and also be frustrated or hurt by them. That's not contradiction; that's being human. What's going on with your family right now that's weighing on you?",
                "Teenage years often mean growing independence while still being under your parents' roof - and that tension is real. They might not get what you're going through, and you might not get why they do what they do. But there's usually communication possible. What's the main tension right now?",
                "Family problems can make home feel less like a safe space, and that's really difficult. Whether it's pressure around expectations, conflicts, feeling unsupported, or just not feeling heard - all of that matters. What's your family situation like? What would help you most?",
                "Sometimes parents or siblings just can't see you the way you see yourself, or they have their own stuff that gets in the way. That doesn't mean you're the problem. What's happening between you and your family? What do you need?",
            ]
            return self._select_varied_response(family_responses, user_id, "family")
        
        # Cultural/Identity/Belonging issues - varied responses
        if any(kw in message_lower for kw in ['culture', 'cultural', 'identity', 'religion', 'faith', 'different', 'fit in', 'belong', 'discrimin', 'racist', 'stereotype', 'heritage', 'tradition', 'accept']):
            cultural_responses = [
                "Figuring out your identity - especially if you're navigating different cultural backgrounds, beliefs, or just feeling like you don't fit in - is genuinely complex work. And doing it as a teenager while everyone's watching? That's a lot. How are you feeling about where you belong? What's making this question come up for you?",
                "Questions about culture, identity, faith, and where you fit in the world are deep and important. You might feel caught between different worlds sometimes. That's not something to fix; it's something to navigate thoughtfully. What's going on that's brought this up?",
                "If you're experiencing discrimination or feeling like you don't belong because of your background, culture, or identity - that's real harm and it's not okay. Your identity matters, and you deserve spaces where you're valued. What's your experience been?",
                "Cultural identity as a teenager can feel like a constant negotiation - between what your family/community values and what the wider world seems to value, or what you personally feel. None of that is simple. What part of your identity feels most complicated or unsupported right now?",
                "Belonging and acceptance are fundamental human needs, and when you don't feel them, it hurts. Whether it's about your culture, your values, your beliefs, or just who you are - you deserve to be seen and accepted. Where are you struggling to feel that right now?",
            ]
            return self._select_varied_response(cultural_responses, user_id, "cultural")
        
        # Teenage emotions/identity/growing up - varied responses
        if any(kw in message_lower for kw in ['angry', 'mood', 'emotion', 'feeling confused', 'identity', 'who am i', 'changing', 'growing up', 'teenage', 'hormones', 'emotional', 'angry at myself', 'irritable']):
            teen_emotion_responses = [
                "Your emotions as a teenager are intense and real - that's not you being 'too emotional' or 'overdramatic.' Your brain is literally rewiring itself right now, and everything feels amplified. That makes sense. What emotions are you struggling with? Just naming them sometimes helps.",
                "The teenage years are when you're building your identity, your values, your sense of self - and that's not a stable process. Some days you might feel one way, other days completely different. That's normal. What emotional stuff are you navigating right now?",
                "Anger, confusion, sadness, joy - all of it can feel so big when you're your age. And sometimes it comes out in ways you don't expect or want. That's okay. You're learning how to process these feelings. What's going on emotionally that brought this up?",
                "Being a teenager means your body's changing, your hormones are all over the place, your brain's developing, AND you're supposed to figure out who you want to be. No wonder emotions feel overwhelming sometimes. What's happening inside that you need to process?",
                "Your emotional life matters - just as much as anyone else's. Don't let anyone tell you that you're being silly or overthinking for having big feelings right now. What are you going through emotionally?",
            ]
            return self._select_varied_response(teen_emotion_responses, user_id, "teen_emotions")
        
        # Stress management/coping skills - varied responses
        if any(kw in message_lower for kw in ['stress manag', 'coping', 'handle', 'deal with', 'cope', 'manage stress', 'overwhelm', 'too much', 'coping skill', 'pressure', 'breaking point']):
            stress_mgmt_responses = [
                "The fact that you're thinking about stress management and coping skills? That's actually really smart and self-aware. You're recognizing you need tools. There are so many real strategies - from breathing techniques to time management to knowing when to reach out. What type of stress management appeals to you most - prevention, in-the-moment relief, or longer-term changes?",
                "Stress is like water - it needs somewhere to go. Without outlets and coping strategies, it backs up and becomes overwhelming. Good news: there are real, practical things that help. Exercise, creative outlets, talking it out, setting boundaries, time management - what sounds useful for your life?",
                "Managing stress isn't about never being stressed. It's about having tools so stress doesn't consume you. Some people need physical release (exercise, sports), some need creative outlets (art, music, writing), some need connection (talking to friends), some need structure. What helps you decompress?",
                "You're carrying a lot right now, and I respect that you're looking for ways to handle it better. That takes wisdom. Real stress management comes down to: what do YOU need to recharge? What calms your nervous system? Let's figure out what actually works for you.",
                "Stress management is a skill you're learning - and that's okay if you don't have it figured out yet. We can build practical strategies together. What does stress feel like for you? How does it show up in your body and mind?",
            ]
            return self._select_varied_response(stress_mgmt_responses, user_id, "stress_management")
        
        # Meditation/mindfulness/relaxation - varied responses
        if any(kw in message_lower for kw in ['meditation', 'mindfulness', 'relaxation', 'calm', 'breathing', 'stress relief', 'coping', 'grounding']):
            meditation_responses = [
                "That's a great instinct - mindfulness and meditation can be genuinely helpful for calming your nervous system. Some different approaches: breathing exercises (like 4-7-8 breathing), body scans (noticing sensations from head to toe), or guided meditations. Apps like Calm or Insight Timer have good structured sessions. What type of relaxation appeals to you most?",
                "Seeking out ways to calm your mind and body is really healthy. Meditation comes in many forms - it doesn't have to be sitting silently. Some people find movement meditation (yoga, tai chi) easier. Others prefer grounding techniques (5 senses exercise) or progressive muscle relaxation. Have you tried any of these before? What felt most natural?",
                "The fact that you're looking into ways to relax tells me you're being proactive about your wellbeing - I respect that. Different techniques work for different people. Breathing exercises, meditation, journaling, even a warm bath - these all signal to your nervous system that you're safe. What resonates with you? What have you found helpful in the past?",
                "Taking time to breathe and center yourself is one of the most powerful things you can do. It literally shifts your nervous system from fight-or-flight to rest-and-digest. Whether it's meditation, deep breathing, progressive muscle relaxation, or any mindfulness practice - these are all excellent tools. Are you looking for guidance on where to start?",
            ]
            return self._select_varied_response(meditation_responses, user_id, "meditation")
        
        # Self-care/wellbeing - varied responses
        if any(kw in message_lower for kw in ['self-care', 'exercise', 'health', 'eating', 'diet', 'fitness', 'energy', 'taking care', 'wellness', 'body image']):
            selfcare_responses = [
                "Taking care of yourself - whether it's moving your body, eating well, sleeping, or doing things you enjoy - is one of the most powerful things you can do for your mental health. These aren't luxuries; they're necessities. What's one self-care thing you could do today that would help? Even something small counts.",
                "When we prioritize our basic needs - moving, eating, resting - everything feels more manageable. It's not about perfection; it's about being gentle with yourself. What's one area of self-care you'd like to focus on? Sometimes starting with one thing makes it easier to build from there.",
                "Your wellbeing matters, and investing in it - through exercise, good food, rest, or things you enjoy - is investing in your mental health too. These are connected. What does good self-care look like for you? What makes you feel better in your body and mind?",
                "Sometimes the most therapeutic thing we can do is the simplest: move our body (even a 10-minute walk), eat something nourishing, get outside, or do something we enjoy. These basics really do make a difference. What's one thing you could do today to take care of yourself?",
            ]
            return self._select_varied_response(selfcare_responses, user_id, "selfcare")
        
        # Default - varied empathetic responses
        default_responses = [
            "I'm hearing that something real is bothering you, and I appreciate you sharing that. What you're feeling matters. Talk me through what's going on - sometimes just naming it out loud helps.",
            "There's something you want to talk about, and I'm here for it. I might not have all the answers, but I can listen and maybe help you figure some things out. What's on your mind?",
            "Something's weighing on you, and that's clear. I'm genuinely interested in understanding what you're dealing with. Take your time and tell me what's going on.",
            "You reached out, which means something needs to shift. I'm listening. What's the most pressing thing you're dealing with right now?",
            "It sounds like you're carrying something difficult. I want to understand what you're experiencing. Can you tell me more about what's happening?",
        ]
        return self._select_varied_response(default_responses, user_id, "default")
    
    def get_bot_info(self) -> Dict:
        """
        Return bot information and capabilities
        
        Returns:
            Dict: Bot metadata
        """
        return {
            'bot_name': 'SafeTalk Bot - LLM Enhanced',
            'version': '4.0-LLM',
            'model': self.model_name if self.model_ready else 'Fallback (Rule-based)',
            'available': self.model_ready,
            'capabilities': [
                'Natural language understanding with LLM',
                'Contextual conversation tracking',
                'Crisis level assessment (0-10)',
                'Evidence-based therapeutic responses',
                'Automatic escalation for high-risk situations',
                'Multi-turn conversation memory',
                'Fallback mode for LLM unavailability',
                'Real-time response generation'
            ],
            'therapeutic_frameworks': [
                'Cognitive Behavioral Therapy (CBT)',
                'Dialectical Behavior Therapy (DBT)',
                'Motivational Interviewing (MI)',
                'Person-Centered Counseling',
                'Acceptance & Commitment Therapy (ACT)'
            ],
            'response_structure': [
                'Reflection - Understanding their experience',
                'Validation - Affirming their emotions',
                'Exploration - Asking clarifying questions',
                'Hope - Providing perspective and support',
                'Action - Suggesting coping strategies'
            ],
            'crisis_detection': {
                'enabled': True,
                'levels': '0-10 scale',
                'auto_escalation': 'severity >= 9',
                'crisis_resources': 'Provided at severity >= 9'
            },
            'model_info': {
                'engine': 'Ollama' if self.model_ready else 'N/A',
                'backend': 'Local self-hosted',
                'latency': 'Depends on hardware',
                'data_privacy': 'All processing local, no external API calls'
            }
        }
