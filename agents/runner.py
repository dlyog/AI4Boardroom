# agents/runner.py

import asyncio
import logging
from agents.core_agents import ALL_AGENTS

async def run_agent_async(agent_name, text):
    """Run an ADK agent asynchronously and return structured result."""
    agent = ALL_AGENTS.get(agent_name)
    if not agent:
        return {
            "agent_name": agent_name,
            "decision": "reject",
            "comments": f"Agent {agent_name} not found",
            "irs_expense_category": None
        }

    try:
        result = await agent.run(text)

        # Normalize and extract expected fields
        decision = result.get("decision", "reject").lower()
        comments = result.get("comments", "")
        irs_category = result.get("irs_expense_category", "Unknown") if agent_name == "Tax Agent" else None

        return {
            "agent_name": agent_name,
            "decision": decision,
            "comments": comments,
            "irs_expense_category": irs_category
        }

    except Exception as e:
        logging.error(f"Error running agent {agent_name}: {e}")
        return {
            "agent_name": agent_name,
            "decision": "reject",
            "comments": "Error running agent via ADK",
            "irs_expense_category": None
        }


def run_agent(agent_name, text):
    """Entry point for synchronous code like Flask threads."""
    return asyncio.run(run_agent_async(agent_name, text))
