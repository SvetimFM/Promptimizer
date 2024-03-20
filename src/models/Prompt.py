from typing import Optional

from pydantic import BaseModel, root_validator


# TODO: Implement 'is_monte_carlo: bool' and its logic
# TODO: Implement optimization for prompts for image generation and code output explicitly
# TODO: CRITICAL - implement sorting for Prompt objects on score

class Prompt(BaseModel):
    gen: int                                    # iteration of the prompt -> 0 is seed prompt
    id_in_gen: int                              # id of the prompt in the generation
    id: str = None                              # id will be set by the root_validator
    val: str                                    # prompt itself

    optimization_vector: Optional[str] = None   # semantic critique of the prompt used to improve it
    test_data: Optional[dict[any, any]] = None  # test data used to evaluate the prompt's quality
    toa: Optional[str] = None                   # target of action used to optimize the prompt
    score: Optional[float] = None               # score of the prompt against test data

    @root_validator(pre=True, allow_reuse=True)  # `pre=True` means this runs before field validation
    def init_id(cls, values):
        gen = values.get('gen')
        id_in_gen = values.get('id_in_gen')
        if gen is not None and id_in_gen is not None:
            values['id'] = f"{gen}_{id_in_gen}"
        return values

    def __str__(self):
        return f"{self.id} {self.val} {self.score}"
