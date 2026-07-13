"""
Assessment calculation utilities
"""
from app.utils.dass21_calculator import DASS21Calculator

class DailyCheckInCalculator:
    """Calculate risk score from daily check-in data"""
    
    @staticmethod
    def calculate(checkin_data: dict) -> float:
        """
        Calculate daily check-in risk score (0-100)
        Lower values = lower risk, Higher values = higher risk
        """
        score = 50  # Baseline
        
        # Mood (1-5) - lower mood = higher risk
        mood = checkin_data.get("mood", 3)
        score += (5 - mood) * 8  # More negative mood = increase score
        
        # Sleep hours - too little or too much = concern
        sleep_hours = checkin_data.get("sleep_hours", 7)
        if sleep_hours < 4 or sleep_hours > 10:
            score += 15
        elif sleep_hours < 6 or sleep_hours > 9:
            score += 10
        
        # Exercise - less exercise = higher risk
        exercise_minutes = checkin_data.get("exercise_minutes", 0)
        if exercise_minutes == 0:
            score += 10
        elif exercise_minutes < 30:
            score += 5
        
        # Social interaction - isolation = higher risk
        social_interaction = checkin_data.get("social_interaction", "Limited")
        social_map = {
            "None": 20,
            "Limited": 10,
            "Moderate": 0,
            "Good": -10
        }
        score += social_map.get(social_interaction, 0)
        
        # Stress level (1-10)
        stress_level = checkin_data.get("stress_level", 5)
        score += (stress_level - 5) * 2
        
        # Anxiety level (1-10)
        anxiety_level = checkin_data.get("anxiety_level", 5)
        score += (anxiety_level - 5) * 2
        
        # Alert flags
        if checkin_data.get("negative_thoughts", False):
            score += 25
        if checkin_data.get("substance_use_today", False):
            score += 15
        if checkin_data.get("self_harm_thoughts", False):
            score += 35
        
        # Normalize to 0-100
        score = max(0, min(100, score))
        return round(score, 2)


class ProfileRiskCalculator:
    """Calculate profile assessment risk score"""
    
    @staticmethod
    def calculate(profile_data: dict) -> float:
        """
        Calculate profile risk score (0-100)
        Based on academic, family, financial, and behavioral factors
        """
        score = 0
        
        # Academic performance
        gpa = profile_data.get("gpa", 3.0)
        if gpa < 2.5:
            score += 15
        elif gpa < 3.0:
            score += 10
        
        # Repeated subjects
        repeated = profile_data.get("repeated_subjects", 0)
        score += repeated * 5
        
        # Attendance
        attendance = profile_data.get("attendance", 100)
        if attendance < 60:
            score += 15
        elif attendance < 80:
            score += 8
        
        # Family relationship
        family_score = profile_data.get("family_relationship_score", 10)
        if family_score < 5:
            score += 20
        elif family_score < 7:
            score += 15
        
        # Income level
        income_level = profile_data.get("income_level", "Medium")
        if income_level == "Low":
            score += 15
        
        # Living arrangement (isolation = risk)
        living = profile_data.get("living_arrangement", "With Family")
        if living == "Alone":
            score += 10
        
        # Employment stress
        employment = profile_data.get("employment_status", "Student")
        if employment == "Full-time + Studies":
            score += 10
        
        # Financial stress
        if profile_data.get("financial_stress", False):
            score += 15
        
        # Communication skills (protective factor)
        comm_skills = profile_data.get("communication_skills", 5)
        if comm_skills >= 4:
            score -= 5
        
        # Social connection (protective factor)
        social_conn = profile_data.get("social_connection", 5)
        if social_conn >= 4:
            score -= 5
        
        # Sleep pattern
        sleep = profile_data.get("sleep_pattern", "Regular")
        if sleep == "Irregular":
            score += 10
        elif sleep == "Very Poor":
            score += 15
        
        # Exercise
        exercise = profile_data.get("exercise_frequency", "Occasionally")
        if exercise == "Never":
            score += 8
        elif exercise == "Rarely":
            score += 5
        
        # Substance use
        substance = profile_data.get("substance_use", "None")
        if substance == "Frequently":
            score += 20
        elif substance == "Occasionally":
            score += 10
        
        # Normalize to 0-100
        score = max(0, min(100, score))
        return round(score, 2)


class AssessmentAggregator:
    """Aggregate multiple assessment modalities into composite score"""
    
    @staticmethod
    def aggregate_with_weights(scores: dict, weights: dict = None) -> dict:
        """
        Aggregate multiple modality scores into composite risk score
        using pre-calculated weights
        
        Args:
            scores: Dictionary of modality scores {modality: score}
            weights: Dictionary of weights for each modality
            
        Returns:
            Dictionary with composite score, risk level, and recommendations
        """
        if weights is None:
            # Default weights from your weights.csv
            weights = {
                "profile_score": 0.03,
                "mood_score": 0.006,
                "dass21_score": 0.489,
                "text_score": 0.238,
                "voice_score": 0.05,
                "face_score": 0.078,
                "behavioral_score": 0.109
            }
        
        composite_score = 0
        for modality, score in scores.items():
            weight = weights.get(modality, 0)
            composite_score += score * weight
        
        composite_score = round(composite_score, 2)
        
        # Determine risk level
        if composite_score < 30:
            risk_level = "LOW"
        elif composite_score < 60:
            risk_level = "MEDIUM"
        elif composite_score < 80:
            risk_level = "HIGH"
        else:
            risk_level = "SEVERE"
        
        return {
            "composite_score": composite_score,
            "risk_level": risk_level,
            "modality_breakdown": scores
        }
