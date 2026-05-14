def get_risk_level(probability):
    if probability >= 0.75:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    else:
        return "Low"