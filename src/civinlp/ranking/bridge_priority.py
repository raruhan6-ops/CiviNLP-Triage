from typing import Dict, Any

class BridgePriorityScorer:
    """
    Computes priority scores for bridges based on element conditions (NBI).
    """
    def __init__(self):
        pass
        
    def calculate_severity(self, element_states: Dict[str, Any]) -> float:
        """
        Calculate severity from element states. 
        Placeholder: assumes element_states contains a 'primary_rating' (0-9).
        Lower rating is worse.
        """
        # NBI rating: 9=Excellent, 0=Failed.
        rating = element_states.get("primary_rating", 9) 
        
        # Invert so high score = high priority
        severity = 9.0 - float(rating)
        return max(0.0, severity)
        
    def score_bridge(self, bridge_record: Dict[str, Any]) -> float:
        states = bridge_record.get("element_states", {})
        # If flat schema, might just look at columns
        # Taking a shortcut for NBI logic
        rating = bridge_record.get("deck_rating", 9) # Example column
        return 9.0 - float(rating)
