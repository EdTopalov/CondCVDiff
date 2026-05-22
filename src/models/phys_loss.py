import torch
import torch.nn as nn
import torch.nn.functional as F

class ElectrochemicalLoss(nn.Module):
    """
    Physically justified loss function for cyclic voltammetry.
    Computes peak heights, their positions, Delta E, and half-wave potential in a differentiable manner.
    """
    def __init__(self, temperature=0.01):
        super().__init__()
        self.temp = temperature

    def extract_soft_peaks(self, current, voltage):
        """
        Diff extraction of areas, peaks
        current, voltage shape: [B, 1, Length]
        """
        w_anodic = F.softmax(current / self.temp, dim=-1)
        
        anodic_height = torch.sum(w_anodic * current, dim=-1)
        anodic_pos = torch.sum(w_anodic * voltage, dim=-1)

        w_cathodic = F.softmax(-current / self.temp, dim=-1)
        
        cathodic_height = torch.sum(w_cathodic * current, dim=-1)
        cathodic_pos = torch.sum(w_cathodic * voltage, dim=-1)

        return anodic_height, cathodic_height, anodic_pos, cathodic_pos

    def extract_areas(self, current):
        anodic_area = torch.sum(F.relu(current), dim=-1)
        cathodic_area = torch.sum(F.relu(-current), dim=-1)
        
        return anodic_area, cathodic_area

    def forward(self, pred_c, true_c, voltage):
        """
        pred_c: generated current [B, 1, 968]
        true_c: true current [B, 1, 968]
        voltage: true voltage [B, 1, 968]
        """
        pred_ah, pred_ch, pred_ap, pred_cp = self.extract_soft_peaks(pred_c, voltage)
        pred_aa, pred_ca = self.extract_areas(pred_c)
        pred_delta_E = torch.abs(pred_ap - pred_cp)
        pred_half_wave = (pred_ap + pred_cp) / 2.0

        true_ah, true_ch, true_ap, true_cp = self.extract_soft_peaks(true_c, voltage)
        true_aa, true_ca = self.extract_areas(true_c)
        true_delta_E = torch.abs(true_ap - true_cp)
        true_half_wave = (true_ap + true_cp) / 2.0

        loss_heights = F.l1_loss(pred_ah, true_ah) + F.l1_loss(pred_ch, true_ch)
        loss_positions = F.l1_loss(pred_ap, true_ap) + F.l1_loss(pred_cp, true_cp)
        loss_areas = F.l1_loss(pred_aa, true_aa) + F.l1_loss(pred_ca, true_ca)
        loss_delta_E = F.l1_loss(pred_delta_E, true_delta_E)
        loss_half_wave = F.l1_loss(pred_half_wave, true_half_wave)
        
        total_physics_loss = (
            1.0 * loss_heights + 
            1.0 * loss_positions + 
            0.5 * loss_areas + 
            1.0 * loss_delta_E + 
            1.0 * loss_half_wave
        )
        
        return total_physics_loss