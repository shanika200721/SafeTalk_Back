"""
DASS21 Assessment Calculation Utilities
Depression, Anxiety, and Stress Scale - 21 items
"""

class DASS21Calculator:
    """Calculate DASS21 scores from responses"""
    
    # DASS21 items mapping (0-based indexing for 21 items)
    # Each subscale has 7 items
    # Responses should be 0-3 for each item
    DEPRESSION_ITEMS = [0, 5, 7, 12, 16, 20]        # Questions 1, 6, 8, 13, 17, 21 (0-based) - 6 items
    ANXIETY_ITEMS = [1, 3, 6, 8, 14, 18, 19]       # Questions 2, 4, 7, 9, 15, 19, 20 (0-based) - 7 items  
    STRESS_ITEMS = [2, 4, 9, 10, 13, 17]           # Questions 3, 5, 10, 11, 14, 18 (0-based) - 6 items
    
    # Severity cutoffs (raw scores * 2)
    DEPRESSION_SEVERITY = {
        "Normal": (0, 9),
        "Mild": (10, 13),
        "Moderate": (14, 20),
        "Severe": (21, 27),
        "Very Severe": (28, 42)
    }
    
    ANXIETY_SEVERITY = {
        "Normal": (0, 7),
        "Mild": (8, 9),
        "Moderate": (10, 14),
        "Severe": (15, 19),
        "Very Severe": (20, 42)
    }
    
    STRESS_SEVERITY = {
        "Normal": (0, 14),
        "Mild": (15, 18),
        "Moderate": (19, 25),
        "Severe": (26, 33),
        "Very Severe": (34, 42)
    }
    
    @staticmethod
    def calculate(responses: list) -> dict:
        """
        Calculate DASS21 scores from 21 responses
        
        Args:
            responses: List of 21 integers (0-3 scale)
            
        Returns:
            Dictionary with depression, anxiety, stress scores and severity levels
        """
        if len(responses) != 21:
            raise ValueError("DASS21 requires exactly 21 responses")
        
        # Calculate raw scores (sum of relevant items)
        depression_score = sum(responses[i] for i in DASS21Calculator.DEPRESSION_ITEMS)
        anxiety_score = sum(responses[i] for i in DASS21Calculator.ANXIETY_ITEMS)
        stress_score = sum(responses[i] for i in DASS21Calculator.STRESS_ITEMS)
        
        # Multiply by 2 to get final scores (standard DASS21 calculation)
        depression_score *= 2
        anxiety_score *= 2
        stress_score *= 2
        
        # Calculate total
        total_score = depression_score + anxiety_score + stress_score
        
        # Determine severity classifications
        depression_severity = DASS21Calculator._get_severity(
            depression_score,
            DASS21Calculator.DEPRESSION_SEVERITY
        )
        anxiety_severity = DASS21Calculator._get_severity(
            anxiety_score,
            DASS21Calculator.ANXIETY_SEVERITY
        )
        stress_severity = DASS21Calculator._get_severity(
            stress_score,
            DASS21Calculator.STRESS_SEVERITY
        )
        
        return {
            "depression_score": depression_score,
            "anxiety_score": anxiety_score,
            "stress_score": stress_score,
            "total_dass21_score": total_score,
            "depression_severity": depression_severity,
            "anxiety_severity": anxiety_severity,
            "stress_severity": stress_severity,
            "responses": responses
        }
    
    @staticmethod
    def _get_severity(score: int, severity_dict: dict) -> str:
        """Get severity level based on score"""
        for severity, (min_score, max_score) in severity_dict.items():
            if min_score <= score <= max_score:
                return severity
        return "Very Severe"
    
    @staticmethod
    def calculate_dass21_risk_score(dass21_data: dict) -> float:
        """
        Convert DASS21 scores to 0-100 risk scale
        Higher DASS21 = Higher risk
        Maximum DASS21 total = 126 (42+42+42)
        """
        total_score = dass21_data.get("total_dass21_score", 0)
        risk_score = (total_score / 126) * 100
        return min(100, round(risk_score, 2))
