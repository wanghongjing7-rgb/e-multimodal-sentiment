"""Task extension points; shared training never imports Q2 or Q3."""
from typing import Any
from e_multimodal_sentiment.models.api import ModelOutput


class TaskHook:
    """Override individual methods to attach task-specific behavior."""
    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Inject masks or transform a batch before forward."""
        return batch

    def postprocess_output(self, batch: dict[str, Any], output: ModelOutput) -> ModelOutput:
        """Attach task-specific diagnostics after prediction."""
        return output
