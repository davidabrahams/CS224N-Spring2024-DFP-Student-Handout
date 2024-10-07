import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    assert self.all_head_size == config.hidden_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = proj.transpose(1, 2)
    return proj
  
  def split_heads(self, x: Tensor):
    """
    assume x is a Tensor with dimensions (batch_size, seq_length, hidden_size)
    Returns a Tensor with dimensions (batch_size, num_attention_heads, seq_length, attention_head_size)
    """
    # Split the last dimension into (num_heads, d_head)
    assert x.dim() == 3
    batch_size, seq_length = x.shape[0], x.shape[1]
    x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
    return x.transpose(1, 2)

  def attention(self, key, query, value, attention_mask):
    # Each attention is calculated following eq. (1) of https://arxiv.org/pdf/1706.03762.pdf.
    # Attention scores are calculated by multiplying the key and query to obtain
    # a score matrix S of size [bs, num_attention_heads, seq_len, seq_len].
    # S[*, i, j, k] represents the (unnormalized) attention score between the j-th and k-th
    # token, given by i-th attention head.
    # Before normalizing the scores, use the attention mask to mask out the padding token scores.
    # Note that the attention mask distinguishes between non-padding tokens (with a value of 0)
    # and padding tokens (with a value of a large negative number).

    # Make sure to:
    # - Normalize the scores with softmax.
    # - Multiply the attention scores with the value to get back weighted values.
    # - Before returning, concatenate multi-heads to recover the original shape:
    #   [bs, seq_len, num_attention_heads * attention_head_size = hidden_size].
    assert key.dim() == query.dim() == value.dim() == 4
    batch_size, seq_length = key.shape[0], key.shape[2]
    assert key.shape == query.shape == value.shape == (batch_size, self.num_attention_heads, seq_length, self.attention_head_size)
    assert attention_mask.shape == (batch_size, 1, 1, seq_length)

    # compute the dot product between the Q vector (which has size attention_head_size) and the K vector (which also has size attention_head_size) for each word in the sequence,
    # for each attention head, for each batch.
    attention_scores = query.matmul(key.transpose(2, 3)) / (self.attention_head_size ** 0.5)
    attention_scores = attention_scores + attention_mask # mask out certain values
    # dimension 2 corresponds to the Qs, dimension 3 corresponds to the Ks
    # softmax along the K dimension, so for each Q, they add to 1.
    attention_weights = F.softmax(attention_scores, dim=3)
    # print(attention_scores.shape, attention_weights.shape, query.matmul(key.transpose(2, 3)).shape)
    assert attention_scores.shape == attention_weights.shape == (batch_size, self.num_attention_heads, seq_length, seq_length)

    # for each seq_length length vector in attention_scores (which sums to 1), multiple by the seq_length vector of hidden_dim values at a given index
    # this is computing for each key, a weighted average of the values.
    attention_weighted_values = attention_weights.matmul(value)
    assert attention_weighted_values.shape == (batch_size, self.num_attention_heads, seq_length, self.attention_head_size)
    return attention_weighted_values.transpose(1, 2).contiguous().view(batch_size, seq_length, self.all_head_size)

    """
    Let's say:
    H = hidden_size
    N = num_heads

    self.query is an HxH matrix. We can think of it as N different Hx(H/N) matrices. Each of those matrices
    is responsible for extracting the length H/N "key" for a given attention head. We compute all N keys (each length H/N)
    in a single matrix multiplication, yielding a matrix with dimension seq_length * H (but we can really think of it as
    seq_length * (H/N) * N)
    """


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = BertSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    This function is applied after the multi-head attention layer or the feed forward layer.
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    # Hint: Remember that BERT applies dropout to the transformed output of each sub-layer,
    # before it is added to the sub-layer input and normalized with a layer norm.

    batch_size, seq_len, hidden_size = input.shape

    output = dense_layer(output)
    output = dropout(output)
    # shape should be the same after applying dense_layer + dropout
    assert output.shape == input.shape
  
    output = output + input
    output = ln_layer(output)
    # shape should be the same after applying dense_layer + dropout
    assert output.shape == input.shape
    return output


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first BERT layer) or from the previous BERT layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
    Each block consists of:
    1. A multi-head attention layer (BertSelfAttention).
    2. An add-norm operation that takes the input and output of the multi-head attention layer.
    3. A feed forward layer.
    4. An add-norm operation that takes the input and output of the feed forward layer.
    """
    # 1. Multi-head attention
    attn_output = self.self_attention(hidden_states, attention_mask)
    
    # 2. Add-norm after attention layer
    # Pass the output of the attention layer and the original input to the add-norm function
    attn_output = self.add_norm(
        input=hidden_states,
        output=attn_output,
        dense_layer=self.attention_dense,
        dropout=self.attention_dropout,
        ln_layer=self.attention_layer_norm
    )

    # Step 3: Feed-forward network
    intermediate_output = self.interm_dense(attn_output)
    intermediate_output = self.interm_af(intermediate_output)

    # Step 4: Add-norm for feed-forward network output
    layer_output = self.add_norm(
        input=attn_output,
        output=intermediate_output,
        dense_layer=self.out_dense,
        dropout=self.out_dropout,
        ln_layer=self.out_layer_norm
    )

    return layer_output



class BertModel(BertPreTrainedModel):
  """
  The BERT model returns the final embeddings for each token in a sentence.
  
  The model consists of:
  1. Embedding layers (used in self.embed).
  2. A stack of n BERT layers (used in self.encode).
  3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # Embedding layers.
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Register position_ids (1, len position emb) to buffer because it is a constant.
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # BERT encoder.
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # [CLS] token transformations.
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids: Tensor):

    batch_size, seq_length = input_ids.shape

    # Step 1: Get word embedding from self.word_embedding
    input_embeds = self.word_embedding(input_ids)
    assert input_embeds.shape == (batch_size, seq_length, self.config.hidden_size), \
        f"Word embeddings shape {input_embeds.shape} should be [batch_size, seq_length, hidden_size]"
    # Step 2: Get position embedding using self.pos_embedding and position_ids
    pos_ids = self.position_ids[:, :seq_length] # shape is (1, seq_length)
    pos_embeds = self.pos_embedding(pos_ids) # shape is (1, seq_length, hidden_size)
    assert pos_embeds.shape == (1, seq_length, self.config.hidden_size), \
        f"Position embeddings shape {pos_embeds.shape} should be [1, seq_length, hidden_size]"

    # Step 3: Get token type embeddings using self.tk_type_embedding
    # Since we are not considering token type, this embedding is a placeholder
    tk_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)
    assert tk_type_embeds.shape == (input_ids.size(0), seq_length, self.config.hidden_size), \
        f"Token type embeddings shape {tk_type_embeds.shape} should be [batch_size, seq_length, hidden_size]"

    # Step 4: Add word, position, and token type embeddings together
    embeddings = input_embeds + pos_embeds + tk_type_embeds

    # Step 5: Apply layer normalization and dropout
    embeddings = self.embed_layer_norm(embeddings)
    embeddings = self.embed_dropout(embeddings)

    assert embeddings.shape == (input_ids.size(0), seq_length, self.config.hidden_size), \
        f"Final embeddings shape {embeddings.shape} should be [batch_size, seq_length, hidden_size]"

    return embeddings


  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # Get the extended attention mask for self-attention.
    # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
    # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
    # (with a value of a large negative number).
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # Pass the hidden states through the encoder layers.
    for i, layer_module in enumerate(self.bert_layers):
      # Feed the encoding from the last bert_layer to the next.
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # Get the embedding for each input token.
    embedding_output = self.embed(input_ids=input_ids)
    # Feed to a transformer (a stack of BertLayers).
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    # Get cls token hidden state.
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
