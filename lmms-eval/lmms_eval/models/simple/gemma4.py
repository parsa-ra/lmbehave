from typing import Optional

from transformers import Gemma4ForConditionalGeneration

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.gemma3 import Gemma3


@register_model("gemma4")
class Gemma4(Gemma3):
    """Gemma 4 adapter using the same evaluation and representation context path as Gemma 3."""

    MODEL_CLASS = Gemma4ForConditionalGeneration

    def __init__(
        self,
        pretrained: str = "google/gemma-4-31B-it",
        revision: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(pretrained=pretrained, revision=revision, **kwargs)
