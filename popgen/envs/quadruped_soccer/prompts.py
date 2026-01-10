ENVIRONMENT_DESCRIPTION = """
Two quadrupedal robots are playing soccer on a field.
The goal is to score a goal by passing the defender of the other team.
The agents can move around the field and interact with the ball.

When you want to describe agent behaviors as code, these are the accessible variables:
- `agent1_position`: A tuple (x, y) representing the position of agent 1.
- `agent2_position`: A tuple (x, y) representing the position of agent 2.
- `ball_position`: A tuple (x, y) representing the position of the ball.
- `agent1_action`: A string representing the action taken by agent 1 (e.g., "move", "kick", "pass").
- `agent2_action`: A string representing the action taken by agent 2 (e.g., "move", "kick", "pass").
- `goal_position`: A tuple (x, y) representing the position of the goal.
"""
