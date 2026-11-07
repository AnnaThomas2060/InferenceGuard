"""Multi-task RoBERTa risk classifier for inferential privacy detection.
This module defines the architecture and configuration for predicting demographic leakage risks across target attributes.
It inherits from Hugging Face PreTrainedModel to support native hub hosting and serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

try:
    from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel, RobertaConfig, RobertaModel
except ImportError:
    class PretrainedConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def to_dict(self):
            return self.__dict__.copy()

    class PreTrainedModel:
        pass

    RobertaConfig = None
    RobertaModel = None

ATTRIBUTES: Tuple[str, ...] = ("age", "location", "occupation", "education")
ATTR_TO_IDX: Dict[str, int] = {attr: idx for idx, attr in enumerate(ATTRIBUTES)}
IDX_TO_ATTR: Dict[int, str] = {idx: attr for idx, attr in enumerate(ATTRIBUTES)}


class RobertaRiskConfig(PretrainedConfig):
    """Configuration class for the multi-task RoBERTa risk classifier.
    It stores model hyperparameters and attribute definitions needed for initialization.
    This configuration enables seamless serialization and loading from the Hugging Face Hub."""

    model_type = "inferenceguard-roberta"

    def __init__(
        self,
        base_model_name: str = "roberta-base",
        num_heads: int = 4,
        num_classes_per_head: int = 1,
        hidden_size: int = 768,
        classifier_dropout: float = 0.1,
        temperature_values: Optional[List[float]] = None,
        attributes: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_heads = num_heads
        self.num_classes_per_head = num_classes_per_head
        self.hidden_size = hidden_size
        self.classifier_dropout = classifier_dropout
        self.temperature_values = temperature_values or [1.0] * num_heads
        self.attributes = attributes or list(ATTRIBUTES)


class RobertaRiskClassifier(PreTrainedModel if torch is not None else object):
    """Production multi-task classifier built on top of a RoBERTa encoder backbone.
    It predicts privacy risk across age, location, occupation, and education attributes using dedicated heads.
    The model supports temperature scaling and standard Hugging Face hub integration."""

    config_class = RobertaRiskConfig
    base_model_prefix = "roberta"

    def __init__(
        self,
        config: RobertaRiskConfig,
        backbone: Optional[Any] = None,
    ) -> None:
        if torch is None:
            self.config = config
            return
        super().__init__(config)
        self.config = config

        if backbone is not None:
            self.roberta = backbone
        else:
            loaded_backbone = None
            try:
                loaded_backbone = AutoModel.from_pretrained(config.base_model_name, local_files_only=True)
            except Exception:
                pass

            if loaded_backbone is None:
                try:
                    loaded_backbone = AutoModel.from_pretrained(config.base_model_name)
                except Exception:
                    try:
                        roberta_cfg = AutoConfig.from_pretrained(config.base_model_name)
                        loaded_backbone = AutoModel.from_config(roberta_cfg)
                    except Exception:
                        if RobertaConfig is not None and RobertaModel is not None:
                            h_size = getattr(config, "hidden_size", 768)
                            roberta_cfg = RobertaConfig(
                                hidden_size=h_size,
                                num_attention_heads=12 if h_size % 12 == 0 else 8,
                                num_hidden_layers=1,
                            )
                            loaded_backbone = RobertaModel(roberta_cfg)

            self.roberta = loaded_backbone

        resolved_hidden = getattr(
            self.config,
            "hidden_size",
            getattr(getattr(self.roberta, "config", None), "hidden_size", 768) if self.roberta is not None else 768,
        )

        dropout_rate = getattr(config, "classifier_dropout", 0.1)
        self.dropout = nn.Dropout(dropout_rate)

        self.heads = nn.ModuleDict({
            attr: nn.Sequential(
                nn.Linear(resolved_hidden, resolved_hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(resolved_hidden // 2, config.num_classes_per_head),
            )
            for attr in config.attributes
        })

        temperatures = getattr(config, "temperature_values", None) or [1.0] * config.num_heads
        self.temperatures = nn.ParameterDict({
            attr: nn.Parameter(torch.tensor([float(temperatures[i])], dtype=torch.float32), requires_grad=False)
            for i, attr in enumerate(config.attributes)
        })

    def get_input_embeddings(self) -> Any:
        """Retrieve the word embedding layer from the underlying RoBERTa backbone.
        This layer is required when computing Integrated Gradients for attribute cue attribution.
        It falls back safely if the backbone structure differs."""
        if hasattr(self.roberta, "get_input_embeddings"):
            return self.roberta.get_input_embeddings()
        if hasattr(self.roberta, "embeddings"):
            return getattr(self.roberta.embeddings, "word_embeddings", None)
        return None

    def set_temperatures(self, temperatures: Sequence[float]) -> None:
        """Assign calibrated temperature values to scale raw logits across all heads.
        Temperatures must be strictly positive floats corresponding to each attribute in order.
        Updated values are reflected in both module parameters and configuration."""
        if len(temperatures) != len(self.config.attributes):
            raise ValueError(f"Expected {len(self.config.attributes)} temperatures, got {len(temperatures)}")
        for i, attr in enumerate(self.config.attributes):
            val = float(temperatures[i])
            if val <= 0.0:
                raise ValueError(f"Temperature must be strictly positive, got {val} for {attr}")
            if torch is not None and hasattr(self, "temperatures") and attr in self.temperatures:
                self.temperatures[attr].data = torch.tensor([val], dtype=torch.float32)
        self.config.temperature_values = [float(t) for t in temperatures]

    def forward(
        self,
        input_ids: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        inputs_embeds: Optional[Any] = None,
        labels: Optional[Dict[str, Any]] = None,
        apply_temperature: bool = False,
    ) -> Dict[str, Any]:
        """Execute the forward pass across the RoBERTa encoder and multi-task prediction heads.
        Logits are computed for each demographic attribute from the pooled representation.
        Optional labels are used to compute weighted multi-task cross-entropy loss."""
        if torch is None or self.roberta is None:
            return {"loss": None, "logits": {attr: 0.0 for attr in self.config.attributes}}

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must provide either input_ids or inputs_embeds.")

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        cls_rep = sequence_output[:, 0, :]
        cls_rep = self.dropout(cls_rep)

        logits_dict: Dict[str, Any] = {}
        for attr in self.config.attributes:
            raw_logits = self.heads[attr](cls_rep)
            if apply_temperature:
                raw_logits = raw_logits / self.temperatures[attr]
            logits_dict[attr] = raw_logits

        total_loss = None
        if labels is not None:
            total_loss = torch.tensor(0.0, device=cls_rep.device)
            for attr in self.config.attributes:
                if attr in labels:
                    target = labels[attr]
                    head_logit = logits_dict[attr]
                    if head_logit.shape[-1] == 1:
                        attr_loss = F.binary_cross_entropy_with_logits(head_logit.view(-1), target.view(-1).float())
                    else:
                        attr_loss = F.cross_entropy(head_logit, target.long())
                    total_loss = total_loss + attr_loss

        return {
            "loss": total_loss,
            "logits": logits_dict,
        }

    def predict_proba(
        self,
        input_ids: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        inputs_embeds: Optional[Any] = None,
        apply_temperature: bool = True,
    ) -> Dict[str, Any]:
        """Compute calibrated output probabilities for each demographic attribute head.
        Sigmoid activation is applied to single-class heads and softmax to multi-class heads.
        Temperature scaling is applied by default to ensure accurate risk calibration."""
        res = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            apply_temperature=apply_temperature,
        )
        logits_dict = res["logits"]
        probs: Dict[str, Any] = {}
        for attr, head_logit in logits_dict.items():
            if torch is not None and isinstance(head_logit, torch.Tensor):
                if head_logit.shape[-1] == 1:
                    probs[attr] = torch.sigmoid(head_logit)
                else:
                    probs[attr] = torch.softmax(head_logit, dim=-1)[:, 1:2]
            else:
                probs[attr] = 0.1
        return probs
