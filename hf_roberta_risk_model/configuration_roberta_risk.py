"""Standalone Hugging Face configuration class for InferenceGuard RoBERTa risk model.
This file can be uploaded directly to the Hugging Face Hub to enable remote code execution.
It provides full compatibility with AutoConfig.from_pretrained when trust_remote_code is set to true."""

from transformers import PretrainedConfig


class RobertaRiskConfig(PretrainedConfig):
    """Configuration class for multi-task RoBERTa risk classifier.
    It encapsulates architecture hyperparameters, head dimensions, and attribute definitions.
    The configuration supports seamless serialization on the Hugging Face Hub."""

    model_type = "inferenceguard-roberta"

    def __init__(
        self,
        base_model_name: str = "roberta-base",
        num_heads: int = 4,
        num_classes_per_head: int = 1,
        hidden_size: int = 768,
        classifier_dropout: float = 0.1,
        temperature_values: list = None,
        attributes: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_heads = num_heads
        self.num_classes_per_head = num_classes_per_head
        self.hidden_size = hidden_size
        self.classifier_dropout = classifier_dropout
        self.temperature_values = temperature_values or [1.0] * num_heads
        self.attributes = attributes or ["age", "location", "occupation", "education"]
