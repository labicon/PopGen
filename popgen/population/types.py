from pydantic import BaseModel


class Behavior(BaseModel):
    """
    The behavior represents a coherent strategy or personality that a SINGLE player might adopt in this environment.
    Behaviors are only guiding the actions of one agent, not a team. It can adapt to the dynamics of the game and the actions of other agents, but should not define these latest.
    """

    name: str
    description: str

    def __str__(self):
        return f"Behavior(name={self.name}, description={self.description})"


class CodeSnippet(BaseModel):
    """
    A code snippet representing a piece of code, such as a policy function.
    """

    code: str
    reasoning: str | None = None
