from pydantic import BaseModel

class Behavior(BaseModel):
    """
    The behavior represents a coherent strategy or personality that a player might adopt in this environment.
    """
    name: str
    description: str

    def __str__(self):
        return f"Behavior(name={self.name}, description={self.description})"