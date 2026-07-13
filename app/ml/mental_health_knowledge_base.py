"""
Mental Health Knowledge Base for Therapeutic Responses
Implements retrieval-augmented generation (RAG) for evidence-based counselor-like responses
Based on cognitive-behavioral therapy, DBT, and clinical psychology principles
"""

class MentalHealthKnowledgeBase:
    """
    Therapeutic knowledge base with mental health intervention strategies
    Maps user concerns -> therapeutic techniques -> evidence-based responses
    """
    
    MENTAL_HEALTH_KB = {
        # Depression interventions
        'depression': {
            'techniques': ['behavioral_activation', 'cognitive_restructuring', 'values_alignment'],
            'responses': [
                "I hear that you're feeling down. This is a common experience. One thing that can help is gently increasing activities you enjoy, even in small ways. What's one small thing you've enjoyed before?",
                "Depression often makes us feel stuck. A helpful approach is to challenge negative thoughts. Is there a thought that's been troubling you that we could look at together?",
                "Living with depression is hard. Sometimes identifying our values - what truly matters to us - can help us move forward. What's something that gives your life meaning?"
            ],
            'follow_ups': [
                "How are you managing sleep and eating? These basics matter for mood.",
                "Have you talked to anyone about how you're feeling?",
                "Would it help to break this down into smaller, manageable steps?"
            ]
        },
        
        # Anxiety interventions
        'anxiety': {
            'techniques': ['grounding', 'breathing', 'exposure', 'thought_challenging'],
            'responses': [
                "Anxiety can feel overwhelming. Let's ground you in the present moment. Can you name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste?",
                "When anxiety strikes, our breathing can help. Try this: breathe in for 4 counts, hold for 4, out for 4. This signals safety to your nervous system.",
                "Anxiety often tells us to avoid what worries us, but facing it gently helps. What's one small step you could take toward what concerns you?"
            ],
            'follow_ups': [
                "Where do you feel this anxiety in your body?",
                "What happens if you sit with this feeling for just a minute?",
                "Have you noticed patterns in when this happens?"
            ]
        },
        
        # Stress/Overwhelm interventions
        'stress': {
            'techniques': ['prioritization', 'boundary_setting', 'self_compassion'],
            'responses': [
                "Feeling overwhelmed is a sign something needs to change. Let's identify what's actually pressing right now. What's the one thing weighing most heavily?",
                "You don't have to handle everything at once. Breaking things into smaller pieces makes them manageable. What's a small, doable step you could take today?",
                "When stressed, we often blame ourselves. What would you say to a friend in this situation? Can you extend that same kindness to yourself?"
            ],
            'follow_ups': [
                "What would help you feel even slightly better right now?",
                "Is there something you can let go of or delegate?",
                "Who could support you with some of this?"
            ]
        },
        
        # Suicidal ideation - CRITICAL
        'suicidal_ideation': {
            'techniques': ['crisis_protocol', 'safety_planning', 'reasons_for_living'],
            'responses': [
                "I'm genuinely concerned about what you've shared. Your safety matters. Please reach out to a crisis service right now or call 988 (Suicide & Crisis Lifeline)",
                "These feelings are temporary, even if they don't feel that way. Have you thought about what's kept you going before?",
                "I want to help you stay safe. Let's create a plan: Who can you call right now? What's one thing that's helped before?"
            ],
            'follow_ups': [
                "Do you have a safety plan with coping strategies?",
                "Who in your life can support you through this?",
                "Would you be willing to contact a crisis line right now?"
            ],
            'crisis_resources': [
                "988 Suicide & Crisis Lifeline (US): Call/Text 988",
                "Crisis Text Line: Text HOME to 741741",
                "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/",
                "National Suicide Prevention Lifeline: 1-800-273-8255"
            ]
        },
        
        # Self-harm urges
        'self_harm': {
            'techniques': ['emotion_regulation', 'distress_tolerance', 'crisis_response'],
            'responses': [
                "I hear that you're in a lot of pain right now. Urges to self-harm often come when emotions feel unbearable. You reached out instead - that's strength.",
                "When the urge is strong, extreme temperature changes can help: hold ice, take a hot shower. These activate your nervous system without harm.",
                "Your pain is real and valid. There are ways to express it safely. Would a journal, artwork, or movement help you express what you're feeling?"
            ],
            'follow_ups': [
                "What does self-harm usually help you with? (distraction, emotion release, etc.)",
                "Can we find an alternative that serves the same purpose?",
                "Have you used any coping skills successfully before?"
            ]
        },
        
        # Loneliness/Isolation
        'loneliness': {
            'techniques': ['connection_building', 'values_activities', 'small_steps'],
            'responses': [
                "Loneliness is painful and surprisingly common. Connection often comes through shared activities or interests. What brought you joy in the past?",
                "You don't need many friends, just genuine ones. Even one meaningful connection can help. Is there someone you could reach out to and spend 15 minutes with?",
                "Sometimes when we're lonely, we withdraw more. Small steps outward - even online communities - can help. What's one person or place you could connect with?"
            ],
            'follow_ups': [
                "What activities would you enjoy with others?",
                "Have you considered joining a group or club?",
                "Would professional support (counselor) help you work through this?"
            ]
        },
        
        # Low self-worth
        'low_self_worth': {
            'techniques': ['self_compassion', 'strengths_focus', 'values_alignment'],
            'responses': [
                "The critical voice in your head doesn't have the final say. Everyone has value, including you, even when you can't see it right now.",
                "Let's notice: what would others say are your strengths? Sometimes we're our harshest judges.",
                "What's something you've overcome before? That took strength. Can you see any of that strength in you now?"
            ],
            'follow_ups': [
                "Where did this belief about yourself come from?",
                "What would help you treat yourself with the same kindness you'd show a friend?",
                "What's one small thing you could acknowledge that you did well today?"
            ]
        },
        
        # Coping and gratitude
        'gratitude': {
            'techniques': ['positive_psychology', 'strength_building', 'resilience'],
            'responses': [
                "I'm glad you reached out. Gratitude, even for small things, can shift perspective. What's one small thing that helped you today?",
                "Building strength isn't about being 'fine' - it's about noticing what works. What helped you yesterday that might help today?",
                "You've survived 100% of your hardest days. That's resilience. What's one thing you learned from getting through difficulties?"
            ],
            'follow_ups': [
                "What are you proud of, even if it seems small?",
                "Who or what gives you hope?",
                "How can you remind yourself of your strength when things get hard?"
            ]
        },
        
        # Help and support seeking
        'support_seeking': {
            'techniques': ['resource_connection', 'empowerment', 'professional_help'],
            'responses': [
                "Reaching out for help is a sign of strength, not weakness. There are people and resources ready to support you.",
                "You don't have to figure this out alone. Whether it's talking to a counselor, trusted friend, or support group, connection helps.",
                "Professional support can really make a difference. Would you like help thinking about what type of support might work best for you?"
            ],
            'follow_ups': [
                "What kind of support do you think would help most?",
                "Have you worked with a counselor or therapist before?",
                "What barriers might make reaching out difficult? Can we problem-solve those?"
            ]
        }
    }
    
    THERAPEUTIC_PRINCIPLES = {
        'empathy': "Validate emotions and show understanding",
        'non_judgment': "Avoid criticism; normalize struggles",
        'hope': "Convey that change is possible",
        'autonomy': "Support their decisions; don't impose",
        'validation': "Acknowledge difficulty without minimizing",
        'action_focus': "Suggest concrete, small steps",
        'safety': "Prioritize safety over everything else"
    }
    
    CONVERSATION_STARTERS = {
        'check_in': [
            "How are you doing today?",
            "What's on your mind right now?",
            "Tell me what brought you here.",
            "How have you been taking care of yourself lately?"
        ],
        'deepening': [
            "Tell me more about that.",
            "How does that make you feel?",
            "What's the hardest part of this for you?",
            "What would help right now?"
        ],
        'closing': [
            "What's one small thing you could do today?",
            "How can I support you further?",
            "Would it help to check in again?",
            "Remember: you've gotten through hard things before. You have strength."
        ]
    }
    
    @staticmethod
    def get_therapeutic_response(intent: str, user_context: str = "") -> dict:
        """
        Retrieve evidence-based therapeutic response for detected intent
        """
        kb = MentalHealthKnowledgeBase.MENTAL_HEALTH_KB
        
        if intent in kb:
            category = kb[intent]
            return {
                'intent': intent,
                'techniques': category['techniques'],
                'response': category['responses'][0],  # Primary response
                'alternatives': category['responses'][1:],
                'follow_ups': category['follow_ups'],
                'is_crisis': intent == 'suicidal_ideation',
                'crisis_resources': category.get('crisis_resources', [])
            }
        
        # Fallback for unknown intents
        return {
            'intent': 'support',
            'techniques': ['active_listening', 'validation'],
            'response': "I'm here to listen and support you. Tell me more about what you're experiencing.",
            'alternatives': [
                "That sounds difficult. How are you managing?",
                "I appreciate you sharing this with me. What would help?"
            ],
            'follow_ups': ["What would support look like?", "How can I help?"],
            'is_crisis': False,
            'crisis_resources': []
        }
    
    @staticmethod
    def create_safety_plan() -> dict:
        """
        Generate a safety planning template for users in crisis
        """
        return {
            'warning_signs': [
                'Increased isolation',
                'Talk of hopelessness',
                'Giving away possessions',
                'Sudden calm after crisis',
                'Talking about death'
            ],
            'internal_coping': [
                'Deep breathing exercises',
                'Grounding techniques',
                'Journaling',
                'Physical activity',
                'Creative expression'
            ],
            'people_to_contact': [
                'Family member',
                'Close friend',
                'Counselor/therapist',
                'Crisis line',
                'Trusted mentor'
            ],
            'coping_with_crisis': [
                '988 Suicide & Crisis Lifeline',
                'Go to nearest emergency room',
                'Call local emergency services',
                'Text crisis line',
                'Reach out to trusted person'
            ],
            'reasons_for_living': [
                '[User to fill: Who/what gives me reasons to continue?]',
                '[User to fill: My strengths]',
                '[User to fill: What I want to accomplish]'
            ]
        }

    @staticmethod
    def detect_crisis_keywords(text: str) -> bool:
        """
        Quick crisis detection based on high-risk keywords
        """
        crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'want to die', 'no point', 
            'hopeless', 'worthless', 'self harm', 'cutting', 'overdose',
            'jump', 'hang', 'rope', 'nothing matters', 'better off dead'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in crisis_keywords)
    
    @staticmethod
    def get_empathy_score(intent: str) -> int:
        """
        Calculate empathy requirement level (1-5)
        Guides response warmth and validation level
        """
        high_empathy_intents = ['suicidal_ideation', 'self_harm', 'depression', 'loneliness']
        medium_empathy_intents = ['anxiety', 'stress', 'low_self_worth']
        
        if intent in high_empathy_intents:
            return 5
        elif intent in medium_empathy_intents:
            return 4
        else:
            return 3
