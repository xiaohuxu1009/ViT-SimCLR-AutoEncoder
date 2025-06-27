import torch
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.07):
    batch_size = z_i.shape[0]
    device = z_i.device

    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, p=2, dim=1)

    sim = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim = sim.masked_fill(mask, -1e4)

    log_sim_probs = F.log_softmax(sim, dim=1)

    targets = torch.arange(batch_size, device=device)
    targets = torch.cat([targets + batch_size, targets])

    loss = -log_sim_probs[torch.arange(2 * batch_size, device=device), targets].mean()

    return loss
