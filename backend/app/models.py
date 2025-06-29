from pydantic import BaseModel


class Game(BaseModel):
    id: int
    date: str
    home_team: str
    away_team: str
    score: str
