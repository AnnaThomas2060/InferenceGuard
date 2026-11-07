"""Standalone Hugging Face model architecture class for InferenceGuard RoBERTa risk classifier.
This module is uploaded directly to Hugging Face Hub repositories to support remote code loading.
It implements the 4-head multi-task demographic risk predictor with calibrated temperature scaling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel

from .configuration_roberta_risk import RobertaRiskConfig


class RobertaRiskClassifier(PreTrainedModel):
    """Multi-task privacy risk classifier built on a RoBERTa encoder backbone.
    It predicts inferential leakage across age, location, occupation, and education attributes.
    The architecture supports native Hugging Face loading via from_pretrained."""

    config_class = RobertaRiskConfig
    base_model_prefix = "roberta"

    def __init__(self, config: RobertaRiskConfig):
        super().__init__(config)
        self.config = config

        try:
            self.roberta = AutoModel.from_pretrained(config.base_model_name, local_files_only=True)
        except Exception:
            try:
                self.roberta = AutoModel.from_pretrained(config.base_model_name)
            except Exception:
                try:
                    roberta_cfg = AutoConfig.from_pretrained(config.base_model_name)
                    self.roberta = AutoModel.from_config(roberta_cfg)
                except Exception:
                    from transformers import RobertaConfig, RobertaModel
                    h_size = getattr(config, "hidden_size", 768)
                    roberta_cfg = RobertaConfig(
                        hidden_size=h_size,
                        num_attention_heads=12 if h_size % 12 == 0 else 8,
                        num_hidden_layers=1,
                    )
                    self.roberta = RobertaModel(roberta_cfg)

        resolved_hidden = getattr(config, "hidden_size", 768)
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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        apply_temperature: bool = False,
    ):
        """Execute model forward pass across encoder and demographic classification heads.
        Outputs include raw or temperature-scaled logits for each attribute head.
        Cross-entropy loss is computed when ground truth labels are provided."""
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

        logits_dict = {}
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

        return {"loss": total_loss, "logits": logits_dict}

    def predict_proba(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        apply_temperature: bool = True,
    ):
        """Compute calibrated probability scores for all demographic target attributes.
        Probabilities are normalized using sigmoid activation and scaled by temperatures.
        This method provides direct risk estimates for downstream guardrail components."""
        res = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            apply_temperature=apply_temperature,
        )
        logits_dict = res["logits"]
        probs = {}
        for attr, head_logit in logits_dict.items():
            if head_logit.shape[-1] == 1:
                probs[attr] = torch.sigmoid(head_logit)
            else:
                probs[attr] = torch.softmax(head_logit, dim=-1)[:, 1:2]
        return probs
