import torch 
from torch.nn import functional as F

import commons


def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2 


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l




def contrastive_loss(z_p_real, z_p_fake_age, z_p_fake_sex, z_p_fake_dialect, z_mask, margin=1.0):

    # Convert to float for numerical stability
    z_p_real = z_p_real.float()
    z_p_fake_age = z_p_fake_age.float()
    z_p_fake_sex = z_p_fake_sex.float()
    z_p_fake_dialect = z_p_fake_dialect.float()
    z_mask = z_mask.float()
    
    total_loss = 0.0
    num_pairs = 0
    
    # All embeddings to compare
    embeddings = {
        'real': z_p_real,
        'fake_age': z_p_fake_age,
        'fake_sex': z_p_fake_sex,
        'fake_dialect': z_p_fake_dialect
    }
    
    # Compute pairwise contrastive losses
    emb_names = list(embeddings.keys())
    for i, name1 in enumerate(emb_names):
        for j, name2 in enumerate(emb_names):
            if i >= j:  # Skip duplicate pairs and self-comparison
                continue
            
            emb1 = embeddings[name1]
            emb2 = embeddings[name2]
            
            # Compute squared euclidean distance
            diff = emb1 - emb2
            squared_dist = torch.sum(diff ** 2, dim=1, keepdim=True)  # [b, 1, t_t]
            
            # Contrastive loss: we want distance >= margin
            # Loss = max(0, margin - distance)^2
            distance = torch.sqrt(squared_dist + 1e-8)  # Add epsilon for numerical stability
            loss = torch.clamp(margin - distance, min=0.0) ** 2

            total_loss += torch.mean(loss)
            num_pairs += 1
    

    l = total_loss / num_pairs
    return l


def contrastive_loss_cosine(z_p_real, z_p_fake_age, z_p_fake_sex, z_p_fake_dialect, z_mask, margin=0.5):
    """
    Contrastive loss using cosine similarity
    
    Args:
        z_p_real: Real embeddings [b, h, t_t]
        z_p_fake_age: Fake age embeddings [b, h, t_t]
        z_p_fake_sex: Fake sex embeddings [b, h, t_t]
        z_p_fake_dialect: Fake dialect embeddings [b, h, t_t]
        z_mask: Mask for valid timesteps [b, 1, t_t] or [b, h, t_t]
        margin: Cosine similarity margin (default: 0.5)
    
    Returns:
        l: Average contrastive loss
    """
    # Convert to float
    z_p_real = z_p_real.float()
    z_p_fake_age = z_p_fake_age.float()
    z_p_fake_sex = z_p_fake_sex.float()
    z_p_fake_dialect = z_p_fake_dialect.float()
    z_mask = z_mask.float()
    
    def normalize_embedding(emb):
        # L2 normalize along the feature dimension
        norm = torch.norm(emb, dim=1, keepdim=True) + 1e-8
        return emb / norm
    
    total_loss = 0.0
    num_pairs = 0
    
    # Normalize all embeddings
    embeddings = {
        'real': normalize_embedding(z_p_real),
        'fake_age': normalize_embedding(z_p_fake_age),
        'fake_sex': normalize_embedding(z_p_fake_sex),
        'fake_dialect': normalize_embedding(z_p_fake_dialect)
    }
    
    # Compute pairwise cosine similarity losses
    emb_names = list(embeddings.keys())
    for i, name1 in enumerate(emb_names):
        for j, name2 in enumerate(emb_names):
            if i >= j:  # Skip duplicate pairs and self-comparison
                continue
            
            emb1 = embeddings[name1]
            emb2 = embeddings[name2]
            
            # Compute cosine similarity
            cosine_sim = torch.sum(emb1 * emb2, dim=1, keepdim=True)  # [b, 1, t_t]
            
            # We want low similarity (negative or close to zero)
            # Loss increases as similarity increases above -margin
            loss = torch.clamp(cosine_sim + margin, min=0.0) ** 2
            
            # Apply mask and accumulate
            
            masked_loss = loss * z_mask
            total_loss += torch.sum(masked_loss)
            num_pairs += 1
    
    # Normalize by number of valid timesteps and number of pairs
    valid_timesteps = torch.sum(z_mask)
    l = total_loss / (valid_timesteps * num_pairs)
    
    return l

def mulsupcon_loss(all_embeddings, all_ages, all_sex, all_dialects, 
                         speaker_embeddings, batch_size, temperature=0.07, 
                         speaker_sim_threshold=0.7):

    
    device = all_embeddings.device
    all_embeddings = all_embeddings.flatten(0, 1)
    all_embeddings = F.normalize(all_embeddings, dim=1)
    

    sim_matrix = torch.mm(all_embeddings, all_embeddings.t()) / temperature
    
    # Compute speaker similarity matrix
    speaker_embeddings_norm = F.normalize(speaker_embeddings, dim=1)
    speaker_sim_matrix = torch.mm(speaker_embeddings_norm, speaker_embeddings_norm.t())
    
    total_loss = 0.0
    num_terms = 0
    
    # Only use original samples as anchors (first batch_size samples)
    for i in range(batch_size):
        anchor_age = all_ages[i]
        anchor_sex = all_sex[i] 
        anchor_dialect = all_dialects[i]
        
        # Get similarities for this anchor
        anchor_sims = sim_matrix[i]
        
        # Exclude self from denominator
        denom_mask = torch.ones_like(anchor_sims, dtype=torch.bool)
        denom_mask[i] = False
        denominator = torch.logsumexp(anchor_sims[denom_mask], dim=0)
        
        # Find positives for each attribute
        attributes = [
            ('age', all_ages == anchor_age),
            ('sex', all_sex == anchor_sex), 
            ('dialect', all_dialects == anchor_dialect)
        ]
        
        for attr_name, attr_mask in attributes:
            # Combine attribute match with speaker similarity
            speaker_sim_mask = speaker_sim_matrix[i] > speaker_sim_threshold
            positive_mask = attr_mask & speaker_sim_mask
            positive_mask[i] = False  # Exclude self
            
            # Get positive indices
            positive_indices = torch.where(positive_mask)[0]
            if len(positive_indices) == 0:
                continue
                
            # Compute loss for this attribute
            positive_sims = anchor_sims[positive_indices]
            attr_loss = (-positive_sims + denominator).mean()
            
            total_loss += attr_loss
            num_terms += 1
    
    return total_loss / max(num_terms, 1) if num_terms > 0 else torch.tensor(0.0, device=device)