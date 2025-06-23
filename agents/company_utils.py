import json
import os

def load_company_profile(company_name: str):
    path = f"companies/{company_name}/company_profile.json"
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def reduce_company_profile(profile: dict) -> str:
    """Extract only critical decision-making context for LLMs."""
    summary = {
        "company_name": profile.get("company_name"),
        "mission": profile.get("mission_statement"),
        "capital": profile.get("initial_capital"),
        "core_focus": profile.get("core_focus"),
        "constraints": profile.get("key_constraints"),
        "decision_criteria": profile.get("decision_criteria"),
        "stage": profile.get("current_stage")
    }
    return json.dumps(summary, indent=2)
