import torch
import os
from transformers import RobertaTokenizerFast, RobertaModel, RobertaForSequenceClassification
import joblib
import math
import torch.nn.functional as F

# Base Model Implementation
class BaseModelWrapper(torch.nn.Module):
    """Wrapper for the base RoBERTa model"""
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Return logits and a dummy projected_features for compatibility
        dummy_features = torch.zeros(logits.size(0), 128).to(logits.device)
        return logits, dummy_features

# Custom Model Implementation
class MultiHeadAttentionPooling(torch.nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(self.head_dim)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.pooling_dense = torch.nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, hidden_size = hidden_states.size()
        normalized_hidden = self.layer_norm(hidden_states)

        q = self.query(normalized_hidden).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(normalized_hidden).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(normalized_hidden).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, -1e4)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, hidden_size)
        output = self.out_proj(context)

        cls_output = output[:, 0, :]
        if attention_mask is not None:
            mask_expanded = attention_mask.squeeze(1).squeeze(1).unsqueeze(-1).expand(output.size()).float()
            sum_embeddings = torch.sum(output * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_output = sum_embeddings / sum_mask
        else:
            mean_output = torch.mean(output, dim=1)
        
        max_output, _ = torch.max(output, dim=1)
        combined_pooling = torch.cat([cls_output, mean_output, max_output], dim=-1)
        final_pooled = self.pooling_dense(combined_pooling)
        return final_pooled, attn_weights

class CustomRobertaWithAttentionContrastive(torch.nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.attention_pooling = MultiHeadAttentionPooling(
            hidden_size=768,
            num_heads=12,
            dropout=dropout_rate
        )
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate * 0.7)
        self.dropout3 = torch.nn.Dropout(dropout_rate * 0.5)
        self.dense1 = torch.nn.Linear(768, 512)
        self.dense2 = torch.nn.Linear(512, 256)
        self.dense3 = torch.nn.Linear(256, 128)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.feature_attention = torch.nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        self.residual_proj1 = torch.nn.Linear(768, 512)
        self.residual_proj2 = torch.nn.Linear(512, 256)
        self.classifier = torch.nn.Linear(128, num_labels)
        self.feature_proj = torch.nn.Linear(256, 128)
        self._init_weights()

    def _init_weights(self):
        for module in [self.dense1, self.dense2, self.dense3, self.classifier,
                     self.feature_proj, self.residual_proj1, self.residual_proj2]:
            if hasattr(module, 'weight'):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        pooled, attention_weights = self.attention_pooling(hidden_states, attention_mask)
        x = self.dropout1(pooled)
        dense1_out = F.gelu(self.bn1(self.dense1(x)))
        residual1 = F.gelu(self.residual_proj1(pooled))
        x = dense1_out + residual1
        x = self.dropout2(x)
        dense2_out = F.gelu(self.bn2(self.dense2(x)))
        residual2 = F.gelu(self.residual_proj2(x))
        features_pre_attention = dense2_out + residual2
        features_unsqueezed = features_pre_attention.unsqueeze(1)
        attended_features, feature_attn_weights = self.feature_attention(
            features_unsqueezed, features_unsqueezed, features_unsqueezed
        )
        attended_features = attended_features.squeeze(1)
        enhanced_features = features_pre_attention + attended_features
        projected_features = F.normalize(self.feature_proj(enhanced_features), p=2, dim=1)
        x = self.dropout3(enhanced_features)
        final_features = F.gelu(self.bn3(self.dense3(x)))
        logits = self.classifier(final_features)
        return logits, projected_features

def load_model_components(model_type="custom"):
    """Load the trained model, tokenizer, and label encoder.
    
    Args:
        model_type (str): Type of model to load ("base" or "custom")
        
    Returns:
        tuple: (model, tokenizer, label_encoder, device)
    """
    output_dir = "./sentiment_model_components"
    base_model_path = os.path.join(output_dir, "base_roberta_model.pt")
    custom_model_path = os.path.join(output_dir, "last_roberta_model.pt")
    tokenizer_save_path = os.path.join(output_dir, "roberta_tokenizer")
    label_encoder_save_path = os.path.join(output_dir, "label_encoder.joblib")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    NUM_LABELS = 7
    
    # Load appropriate model based on type
    if model_type == "base":
        model = BaseModelWrapper("roberta-base", NUM_LABELS).to(device)
        model_path = base_model_path
        model_name = "Base RoBERTa"
    else:  # default to custom
        model = CustomRobertaWithAttentionContrastive("roberta-base", NUM_LABELS).to(device)
        model_path = custom_model_path
        model_name = "Custom RoBERTa"
    
    # Load model state dict if file exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")
    
    model.eval()
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_save_path)
    label_encoder = joblib.load(label_encoder_save_path)
    
    return model, tokenizer, label_encoder, device, model_name