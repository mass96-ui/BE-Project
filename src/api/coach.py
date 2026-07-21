from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ScoreInput(BaseModel):
    activation_score: float


@router.post("/coach")
def coach(data: ScoreInput):

    score = data.activation_score

    if score < 40:
        recommendation = """
        Low Soleus Activation

        • Increase resistance by 10%
        • Increase training volume
        • Focus on calf raises
        • Improve ankle stability
        """

    elif score < 70:
        recommendation = """
        Moderate Soleus Activation

        • Maintain current resistance
        • Continue progressive overload
        • Monitor fatigue levels
        """

    else:
        recommendation = """
        High Soleus Activation

        • Reduce resistance slightly
        • Prioritize recovery
        • Include mobility work
        • Avoid overtraining
        """

    return {
        "activation_score": score,
        "recommendation": recommendation
    }