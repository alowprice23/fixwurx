import specialized_agents

# Test each agent type can be instantiated
print("Testing instantiation of all agent types:")

# Test Observer Agent
observer = specialized_agents.ObserverAgent()
print(f"Observer Agent: {observer}")

# Test Analyst Agent
analyst = specialized_agents.AnalystAgent()
print(f"Analyst Agent: {analyst}")

# Test Verifier Agent
verifier = specialized_agents.VerifierAgent()
print(f"Verifier Agent: {verifier}")

# Test Planner Agent
planner = specialized_agents.PlannerAgent()
print(f"Planner Agent: {planner}")

print("\nTesting system prompts:")
print(f"Observer prompt length: {len(specialized_agents._OBSERVER_PROMPT)}")
print(f"Analyst prompt length: {len(specialized_agents._ANALYST_PROMPT)}")
print(f"Verifier prompt length: {len(specialized_agents._VERIFIER_PROMPT)}")
print(f"Planner prompt length: {len(specialized_agents._PLANNER_PROMPT)}")

print("\nSpecialized Agents tests completed successfully")
