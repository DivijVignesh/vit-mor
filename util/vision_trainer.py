import torch
from util.trainer_pt import MoRTrainer

class MoRVisionTrainer(MoRTrainer):
    """Vision-specific trainer for MoR-ViT"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Determine batch size
        if num_items_in_batch is None:
            num_items_in_batch = inputs["pixel_values"].size(0)

        # Forward pass
        outputs = model(
            pixel_values=inputs["pixel_values"],
            num_items_in_batch=num_items_in_batch,
            labels=inputs["labels"]
        )

        # Main classification loss
        loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=outputs.logits.device)

        # Routing losses (fill zeros if None)
        sl  = outputs.sampling_loss     if outputs.sampling_loss     is not None else torch.tensor(0.0, device=loss.device)
        sa  = outputs.sampling_acc      if outputs.sampling_acc      is not None else torch.tensor(0.0, device=loss.device)
        sta = outputs.sampling_topk_acc if outputs.sampling_topk_acc is not None else torch.tensor(0.0, device=loss.device)
        uni = outputs.uniformity        if outputs.uniformity        is not None else torch.tensor(0.0, device=loss.device)
        dead= outputs.dead_token_seq    if outputs.dead_token_seq    is not None else torch.tensor(0.0, device=loss.device)
        bl  = outputs.balancing_loss    if outputs.balancing_loss    is not None else torch.tensor(0.0, device=loss.device)
        br  = outputs.balancing_ratio   if outputs.balancing_ratio   is not None else torch.tensor(0.0, device=loss.device)
        rz  = outputs.router_z_loss     if outputs.router_z_loss     is not None else torch.tensor(0.0, device=loss.device)

        # Combine main loss with weighted routing losses
        if self.cfg.mor.get('enable'):
            if self.cfg.mor.type == "expert":
                loss = loss + self.cfg.mor.expert.coeff * sl + (self.cfg.mor.z_coeff * rz if self.cfg.mor.z_loss else 0)
            else:
                loss = loss + self.cfg.mor.token.coeff * bl + (self.cfg.mor.z_coeff * rz if self.cfg.mor.z_loss else 0)

        # Return the full tuple for Trainer
        out = (loss, sl, sa, sta, uni, dead, bl, br, rz)
        return (out, outputs) if return_outputs else out
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform a prediction step for vision tasks
        """
        # Ensure we have the right input format
        if "pixel_values" not in inputs:
            raise ValueError("Vision trainer expects 'pixel_values' in inputs")
            
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
