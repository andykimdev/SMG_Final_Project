"""
Graph-aware Transformer for Disease-Specific Survival (DSS) prediction.
Uses protein expression + clinical + genomic features with STRING PPI network structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config


class SurvivalGraphTransformer(nn.Module):
    """
    Graph-aware transformer that incorporates PPI network structure
    for survival prediction using Cox Proportional Hazards model.
    
    Extends the classification architecture to handle multimodal inputs
    and predict continuous risk scores.
    """
    
    def __init__(
        self,
        n_proteins: int,
        n_clinical: int,
        n_genomic: int,
        diffusion_kernel: torch.Tensor,
        positional_encodings: torch.Tensor,
        embedding_dim: int = None,
        n_layers: int = None,
        n_heads: int = None,
        dropout: float = None,
        use_clinical: bool = True,
        use_genomic: bool = True,
    ):
        """
        Args:
            n_proteins: Number of protein features (N)
            n_clinical: Number of clinical features
            n_genomic: Number of genomic features
            diffusion_kernel: Graph diffusion kernel K (N×N)
            positional_encodings: Graph positional encodings PE (N×k)
            embedding_dim: Dimension of token embeddings
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            dropout: Dropout probability
            use_clinical: Whether to use clinical features
            use_genomic: Whether to use genomic features
        """
        super().__init__()
        
        # Use config defaults if not specified
        self.embedding_dim = embedding_dim or config.MODEL['embedding_dim']
        self.n_layers = n_layers or config.MODEL['n_layers']
        self.n_heads = n_heads or config.MODEL['n_heads']
        self.dropout = dropout or config.MODEL['dropout']
        self.ffn_dim = config.MODEL['ffn_dim']
        
        self.n_proteins = n_proteins
        self.n_clinical = n_clinical
        self.n_genomic = n_genomic
        self.use_clinical = use_clinical
        self.use_genomic = use_genomic
        
        # Register graph features as buffers (not parameters)
        self.register_buffer('K', diffusion_kernel)  # (N, N)
        self.register_buffer('PE', positional_encodings)  # (N, k)
        
        # ====================================================================
        # Protein token embedding components (same as classification model)
        # ====================================================================
        self.value_projection = nn.Linear(1, self.embedding_dim)  # Project scalar expression value
        self.protein_embedding = nn.Embedding(n_proteins, self.embedding_dim)  # Learned protein-ID embeddings
        self.pe_projection = nn.Linear(positional_encodings.shape[1], self.embedding_dim)  # Project graph PE
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        
        # Graph attention bias (learnable per head)
        self.graph_bias_scale = nn.Parameter(
            torch.ones(self.n_heads) * config.MODEL['graph_bias_scale']
        )
        
        # ====================================================================
        # Clinical feature embedding
        # ====================================================================
        if use_clinical and n_clinical > 0:
            self.clinical_projection = nn.Sequential(
                nn.Linear(n_clinical, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            # Clinical token (like CLS)
            self.clinical_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        else:
            self.clinical_projection = None
            self.clinical_token = None
        
        # ====================================================================
        # Genomic feature embedding
        # ====================================================================
        if use_genomic and n_genomic > 0:
            self.genomic_projection = nn.Sequential(
                nn.Linear(n_genomic, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            # Genomic token (like CLS)
            self.genomic_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        else:
            self.genomic_projection = None
            self.genomic_token = None
        
        # ====================================================================
        # Transformer encoder (same as classification model)
        # ====================================================================
        self.transformer = nn.ModuleList([
            GraphTransformerLayer(
                d_model=self.embedding_dim,
                n_heads=self.n_heads,
                ffn_dim=self.ffn_dim,
                dropout=self.dropout,
            ) for _ in range(self.n_layers)
        ])
        
        self.norm = nn.LayerNorm(self.embedding_dim)
        
        # ====================================================================
        # Risk prediction head (replaces classification head)
        # Outputs single continuous risk score for Cox model
        # ====================================================================
        # Count number of special tokens (CLS + clinical + genomic)
        n_special_tokens = 1  # CLS always present
        if self.clinical_token is not None:
            n_special_tokens += 1
        if self.genomic_token is not None:
            n_special_tokens += 1
        
        self.risk_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim * n_special_tokens, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim // 2, 1)  # Single risk score (no activation)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights (same as classification model)"""
        # Initialize embeddings
        nn.init.normal_(self.protein_embedding.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
        if self.clinical_token is not None:
            nn.init.normal_(self.clinical_token, std=0.02)
        if self.genomic_token is not None:
            nn.init.normal_(self.genomic_token, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass for survival prediction.
        
        Args:
            batch: Dictionary containing:
                   - 'protein': Protein expression values (batch_size, n_proteins)
                   - 'clinical': Clinical features (batch_size, n_clinical)
                   - 'genomic': Genomic features (batch_size, n_genomic)
            
        Returns:
            risk_scores: Predicted risk scores (batch_size, 1)
                        Higher score = higher risk of death
        """
        x = batch['protein']
        batch_size = x.shape[0]
        
        # ====================================================================
        # Create protein token embeddings (same as classification model)
        # ====================================================================
        # 1. Project expression values
        value_emb = self.value_projection(x.unsqueeze(-1))  # (B, N, d)
        
        # 2. Add protein ID embeddings
        protein_ids = torch.arange(self.n_proteins, device=x.device)
        protein_emb = self.protein_embedding(protein_ids)  # (N, d)
        protein_emb = protein_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, d)
        
        # 3. Add graph positional encodings
        pe_emb = self.pe_projection(self.PE)  # (N, d)
        pe_emb = pe_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, d)
        
        # Combine all embeddings
        protein_tokens = value_emb + protein_emb + pe_emb  # (B, N, d)
        
        # ====================================================================
        # Create special tokens for multimodal integration
        # ====================================================================
        special_tokens = []
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d)
        special_tokens.append(cls_tokens)
        
        # Add clinical token if available
        if self.clinical_projection is not None:
            clinical_emb = self.clinical_projection(batch['clinical'])  # (B, d)
            clinical_tokens = self.clinical_token.expand(batch_size, -1, -1)  # (B, 1, d)
            clinical_tokens = clinical_tokens + clinical_emb.unsqueeze(1)  # Modulate with clinical features
            special_tokens.append(clinical_tokens)
        
        # Add genomic token if available
        if self.genomic_projection is not None:
            genomic_emb = self.genomic_projection(batch['genomic'])  # (B, d)
            genomic_tokens = self.genomic_token.expand(batch_size, -1, -1)  # (B, 1, d)
            genomic_tokens = genomic_tokens + genomic_emb.unsqueeze(1)  # Modulate with genomic features
            special_tokens.append(genomic_tokens)
        
        # Concatenate: [CLS, clinical?, genomic?, protein_1, ..., protein_N]
        special_tokens.extend([protein_tokens])
        tokens = torch.cat(special_tokens, dim=1)  # (B, n_special + N, d)
        
        # ====================================================================
        # Create graph attention bias (extended for special tokens)
        # ====================================================================
        graph_bias = self._create_graph_attention_bias(batch_size)  # (B, n_heads, seq_len, seq_len)
        
        # ====================================================================
        # Apply transformer layers (same as classification model)
        # ====================================================================
        for layer in self.transformer:
            tokens = layer(tokens, graph_bias)
        
        tokens = self.norm(tokens)
        
        # ====================================================================
        # Extract special token representations and predict risk
        # ====================================================================
        # Number of special tokens
        n_special = 1  # CLS
        if self.clinical_token is not None:
            n_special += 1
        if self.genomic_token is not None:
            n_special += 1
        
        # Extract all special token embeddings
        special_outputs = tokens[:, :n_special, :]  # (B, n_special, d)
        
        # Flatten and predict risk
        special_outputs_flat = special_outputs.reshape(batch_size, -1)  # (B, n_special * d)
        risk_scores = self.risk_predictor(special_outputs_flat)  # (B, 1)
        
        return risk_scores
    
    def _create_graph_attention_bias(self, batch_size: int) -> torch.Tensor:
        """
        Create attention bias matrix incorporating graph structure.
        Extended to handle clinical and genomic tokens.
        
        Args:
            batch_size: Current batch size
            
        Returns:
            bias: Attention bias (batch_size, n_heads, seq_len, seq_len)
        """
        device = self.K.device
        
        # Count special tokens
        n_special = 1  # CLS
        if self.clinical_token is not None:
            n_special += 1
        if self.genomic_token is not None:
            n_special += 1
        
        seq_len = n_special + self.n_proteins  # Special tokens + proteins
        
        # Initialize bias matrix (all zeros)
        bias = torch.zeros(self.n_heads, seq_len, seq_len, device=device)
        
        # Add graph structure bias ONLY for protein-protein attention
        # Special tokens (CLS, clinical, genomic) have no graph bias
        for h in range(self.n_heads):
            # Add scaled diffusion kernel to protein-protein block
            bias[h, n_special:, n_special:] = self.graph_bias_scale[h] * self.K
        
        # Expand for batch dimension
        bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, n_heads, seq_len, seq_len)
        
        return bias


class GraphAwareMultiheadAttention(nn.Module):
    """
    Custom multi-head attention that properly applies graph structure bias.
    
    NOTE FOR REPORT: This replaces PyTorch's MultiheadAttention to enable
    graph-aware attention via the STRING PPI diffusion kernel.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attn_bias: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            attn_bias: Attention bias (batch_size, n_heads, seq_len, seq_len)
                      Added to attention logits before softmax
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        B, L, D = x.shape
        
        # Project to Q, K, V and reshape for multi-head attention
        # (B, L, D) -> (B, L, n_heads, d_head) -> (B, n_heads, L, d_head)
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute attention scores
        # (B, n_heads, L, d_head) @ (B, n_heads, d_head, L) -> (B, n_heads, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # **KEY FIX: Add graph bias to attention scores**
        if attn_bias is not None:
            scores = scores + attn_bias
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (B, n_heads, L, L) @ (B, n_heads, L, d_head) -> (B, n_heads, L, d_head)
        output = torch.matmul(attn_weights, V)
        
        # Reshape back to (B, L, D)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        
        # Final output projection
        output = self.out_proj(output)
        
        return output


class GraphTransformerLayer(nn.Module):
    """Single transformer layer with graph-aware attention."""
    
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Use custom graph-aware attention instead of PyTorch's MultiheadAttention
        self.self_attn = GraphAwareMultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with graph-aware attention bias.
        
        Args:
            x: Input tokens (batch_size, seq_len, d_model)
            attn_bias: Attention bias (batch_size, n_heads, seq_len, seq_len)
                      Properly applied to incorporate PPI network structure
            
        Returns:
            Output tokens (batch_size, seq_len, d_model)
        """
        # Self-attention with residual (pre-norm)
        residual = x
        x = self.norm1(x)
        
        # Graph-aware attention with bias actually applied
        attn_output = self.self_attn(x, attn_bias=attn_bias)
        x = residual + self.dropout1(attn_output)
        
        # FFN with residual (pre-norm)
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.ffn(x))
        
        return x


# ============================================================================
# Cox Proportional Hazards Loss
# ============================================================================

class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards partial likelihood loss for censored survival data.
    
    Loss = -log(partial likelihood)
         = -sum_{i in events} [risk_i - log(sum_{j in risk_set_i} exp(risk_j))]
    
    This handles right-censored data properly.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, risk_scores: torch.Tensor, time: torch.Tensor,
                event: torch.Tensor) -> torch.Tensor:
        """
        Compute Cox partial likelihood loss.

        Args:
            risk_scores: Predicted risk scores (batch_size, 1) or (batch_size,)
            time: Survival times (batch_size,)
            event: Event indicators (batch_size,) - 1=death, 0=censored

        Returns:
            loss: Cox partial likelihood loss (scalar)
        """
        # Ensure correct shapes - squeeze only the last dimension if 2D
        if risk_scores.dim() == 2:
            risk_scores = risk_scores.squeeze(-1)  # (batch_size, 1) -> (batch_size,)
        elif risk_scores.dim() == 0:
            # Handle scalar case (batch_size=1)
            risk_scores = risk_scores.unsqueeze(0)  # () -> (1,)
        
        # Sort by time (descending) for efficient risk set computation
        sorted_indices = torch.argsort(time, descending=True)
        time_sorted = time[sorted_indices]
        event_sorted = event[sorted_indices]
        risk_sorted = risk_scores[sorted_indices]
        
        # Compute log risk for each sample
        log_risk = risk_sorted
        
        # Compute cumulative sum of exp(risk) for risk sets
        # risk_set[i] = sum_{j: time[j] >= time[i]} exp(risk[j])
        exp_risk = torch.exp(risk_sorted)
        
        # Reverse cumsum gives us the risk set sum for each time point
        # (cumsum in descending time order)
        risk_set_sum = torch.cumsum(exp_risk, dim=0)
        
        # Log of risk set sum
        log_risk_set = torch.log(risk_set_sum + 1e-8)  # Add small epsilon for stability
        
        # Partial log-likelihood for events only
        # For each event, subtract log(risk_set_sum) from log_risk
        event_mask = event_sorted > 0
        
        if event_mask.sum() == 0:
            # No events in batch - return zero loss
            return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
        
        partial_log_likelihood = (log_risk - log_risk_set)[event_mask].sum()
        
        # Return negative log-likelihood (we want to minimize)
        loss = -partial_log_likelihood / event_mask.sum()  # Average over events
        
        return loss


class ConcordanceIndex(nn.Module):
    """
    C-index (Harrell's concordance index) for survival prediction.
    
    C-index measures the fraction of all pairs of subjects whose predictions 
    are correctly ordered with respect to their survival times.
    
    C-index = 1.0: Perfect predictions
    C-index = 0.5: Random predictions
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, risk_scores: torch.Tensor, time: torch.Tensor,
                event: torch.Tensor) -> torch.Tensor:
        """
        Compute concordance index.

        Args:
            risk_scores: Predicted risk scores (batch_size,)
            time: Survival times (batch_size,)
            event: Event indicators (batch_size,) - 1=death, 0=censored

        Returns:
            c_index: Concordance index (scalar between 0 and 1)
        """
        # Ensure correct shapes - squeeze only the last dimension if 2D
        if risk_scores.dim() == 2:
            risk_scores = risk_scores.squeeze(-1)  # (batch_size, 1) -> (batch_size,)
        elif risk_scores.dim() == 0:
            # Handle scalar case (batch_size=1)
            risk_scores = risk_scores.unsqueeze(0)  # () -> (1,)
        
        # Convert to numpy for easier computation
        risk_np = risk_scores.detach().cpu().numpy()
        time_np = time.cpu().numpy()
        event_np = event.cpu().numpy()
        
        # Count concordant and discordant pairs
        concordant = 0
        total_pairs = 0
        
        n = len(risk_np)
        for i in range(n):
            if event_np[i] == 0:
                continue  # Skip censored as first in pair
            
            for j in range(n):
                if i == j:
                    continue
                
                # Only consider pairs where i had event and j survived longer
                # or j is censored with time >= time[i]
                if time_np[j] > time_np[i] or (event_np[j] == 0 and time_np[j] >= time_np[i]):
                    total_pairs += 1
                    # Concordant if higher risk died first
                    if risk_np[i] > risk_np[j]:
                        concordant += 1
        
        if total_pairs == 0:
            return torch.tensor(0.5, device=risk_scores.device)
        
        c_index = concordant / total_pairs
        return torch.tensor(c_index, device=risk_scores.device)


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_survival_epoch(model, dataloader, optimizer, device):
    """Train for one epoch with Cox loss."""
    model.train()
    cox_loss = CoxPHLoss()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        risk_scores = model(batch)
        
        # Compute loss
        loss = cox_loss(risk_scores, batch['time'], batch['event'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAINING['grad_clip'])
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_survival(model, dataloader, device):
    """Evaluate survival model with C-index."""
    model.eval()
    cox_loss = CoxPHLoss()
    c_index_metric = ConcordanceIndex()
    
    all_risks = []
    all_times = []
    all_events = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            risk_scores = model(batch)
            loss = cox_loss(risk_scores, batch['time'], batch['event'])
            
            all_risks.append(risk_scores.cpu())
            all_times.append(batch['time'].cpu())
            all_events.append(batch['event'].cpu())
            
            total_loss += loss.item()
            num_batches += 1
    
    # Concatenate all batches
    all_risks = torch.cat(all_risks)
    all_times = torch.cat(all_times)
    all_events = torch.cat(all_events)
    
    # Compute C-index
    c_index = c_index_metric(all_risks, all_times, all_events)
    
    avg_loss = total_loss / num_batches
    
    return avg_loss, c_index.item()