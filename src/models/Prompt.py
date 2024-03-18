from typing import Optional

from pydantic import BaseModel, root_validator


# TODO: Implement 'is_monte_carlo: bool' and its logic
# TODO: Implement optimization for prompts for image generation and code output explicitly
class Prompt(BaseModel):
    gen:        int     # iteration of the prompt -> 0 is seed prompt
    id_in_gen:  int
    id:         str     # f"{gen}_{id_in_gen}"
    val:        str     # prompt itself
    score:      Optional[float]   # score of the prompt against test data

    @root_validator(pre=True)  # `pre=True` means this runs before field validation
    def init_id(self, values):
        self.id: str = f"{self.gen}_{self.id_in_gen}"
        return values

    def __str__(self):
        return f"{self.id} {self.val} {self.score}"
