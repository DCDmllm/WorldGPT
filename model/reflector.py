import torch
from torch import nn
from .qformer import BertLMHeadModel, BertConfig
# from header import *

class Reflector(nn.Module):
    """Layers used in mapping text embeddings to visual outputs."""

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2, cross_attention_freq=1):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.num_hidden_layers = num_hidden_layers
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(self, in_dim: int, out_dim: int, num_output_tokens: int = 16):
        """
        :param mode: ['linear', 'transformer', 'qformer']
        :param freeze_qformer: whether freeze the weights of qformer
        """
        super().__init__()

        self.num_output_tokens = num_output_tokens
        self.out_dim = out_dim

        hidden_dim = 768
        self.fc1 = nn.Linear(in_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(in_dim, hidden_dim)
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_output_tokens, hidden_dim
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, state0: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
            :param state0: (Tensor (B, in_dim)): modality-average languagebind embeddings of state0
            :param action: (Tensor (B, in_dim)): languagebind embeddings of action
            :param context: (Tensor (B, 3, in_dim)): retrieved context (state0, action, state1)
        """
        outputs = None

        x = torch.concat([state0, action], dim=-1)
        x = self.fc1(x).unsqueeze(1)
        context = self.fc2(context)
        image_atts = torch.ones(context.size()[:-1], dtype=torch.long).to(context.device)
        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1) + x.expand(-1, self.query_tokens.shape[1], -1)
        outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=context,
            encoder_attention_mask=image_atts,
            return_dict=True,
        ).last_hidden_state
        outputs = self.fc3(outputs)

        assert outputs.shape[1] == 1 or (outputs.shape[1] * outputs.shape[2] == self.num_output_tokens * self.out_dim), (
        outputs.shape, self.num_output_tokens)
        return outputs  # (B, Q, D)

