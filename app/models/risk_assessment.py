import numpy as np
import pandas as pd
from datetime import datetime
import os

class RiskAssessor:
    def __init__(self):
        # Load weights
        weights_path = os.path.join(os.path.dirname(__file__), 'weights.csv')
        self.weights_df = pd.read_csv(weights_path)
        self.weights = dict(zip(self.weights_df['Modality'], self.weights_df['Weight']))
        
    def calculate_composite_score(self, modality_scores):
        """
        Calculate composite risk score using your derived weights
        
        modality_scores: dict with keys matching your modalities
        Example: {
            'profile_score': 25,
            'mood_score': 30,
            'dass21_score': 65,
            'text_score': 40,
            'voice_score': 35,
            'face_score': 30,
            'behavioral_score': 20
        }
        """
        composite_score = 0
        for modality, score in modality_scores.items():
            weight = self.weights.get(modality, 0)
            composite_score += score * weight
        
        return composite_score
    
    def get_risk_level(self, composite_score):
        """Determine risk level from composite score"""
        if composite_score < 30:
            return "LOW"
        elif composite_score < 60:
            return "MEDIUM"
        elif composite_score < 80:
            return "HIGH"
        else:
            return "SEVERE"
    
    def needs_escalation(self, risk_level):
        """Check if counselor should be notified"""
        return risk_level in ["HIGH", "SEVERE"]
    
    def get_recommendations(self, risk_level, profile_data=None):
        """Generate personalized recommendations"""
        
        base_recommendations = {
            "LOW": [
                "Continue daily mood tracking",
                "Practice mindfulness for 5 minutes daily",
                "Connect with friends and family",
                "Maintain healthy sleep schedule (7-9 hours)",
                "Take breaks between study sessions"
            ],
            "MEDIUM": [
                "Schedule a check-in with university counselor",
                "Try guided breathing exercises (we can help)",
                "Join a student support group",
                "Limit social media, especially before bed",
                "Consider light exercise like walking",
                "Reach out to one trusted friend this week"
            ],
            "HIGH": [
                "⚠️ URGENT: Please contact a counselor within 24 hours",
                "Do not isolate yourself - reach out to someone you trust",
                "Avoid alcohol or other substances",
                "Use crisis hotline if thoughts become overwhelming",
                "We have notified your counselor to check on you",
                "Emergency contact: 1333 (Sri Lanka Mental Health Helpline)"
            ],
            "SEVERE": [
                "🚨 IMMEDIATE ACTION REQUIRED",
                "Counselor has been automatically notified",
                "Do NOT stay alone - contact emergency contact NOW",
                "Call 1333 (Sri Lanka National Mental Health Helpline) immediately",
                "Go to the nearest hospital emergency room",
                "Someone will contact you within 5 minutes"
            ]
        }
        
        # Add personalized elements based on profile if available
        recommendations = base_recommendations[risk_level]
        
        if profile_data:
            if profile_data.get('living_arrangement') == 'Alone' and risk_level in ["HIGH", "SEVERE"]:
                recommendations.append("🏠 Since you live alone, please check in with a friend today")
            
            if profile_data.get('financial_stress'):
                recommendations.append("💰 Financial stress is tough - university has support services")
        
        return recommendations
    
    def assess(self, modality_scores, profile_data=None):
        """Complete risk assessment"""
        
        # Calculate composite score
        composite_score = self.calculate_composite_score(modality_scores)
        
        # Get risk level
        risk_level = self.get_risk_level(composite_score)
        
        # Generate recommendations
        recommendations = self.get_recommendations(risk_level, profile_data)
        
        return {
            'composite_score': round(composite_score, 2),
            'risk_level': risk_level,
            'needs_escalation': self.needs_escalation(risk_level),
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'modality_breakdown': modality_scores
        }

# Create a singleton instance
risk_assessor = RiskAssessor()