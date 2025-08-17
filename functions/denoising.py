import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None):
    with torch.no_grad():
        #setup vectors used in the algorithm
        singulars = H_funcs.singulars()
        Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
        Sigma[:singulars.shape[0]] = singulars
        U_t_y = H_funcs.Ut(y_0)
        Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

        #initialize x_T as given in the paper
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)
        inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device)
        inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
        inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)     

        # implement p(x_T | x_0, y) as given in the paper
        # if eigenvalue is too small, we just treat it as zero (only for init) 
        init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
        init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
        init_y = init_y.view(*x.size())
        remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
        remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
        init_y = init_y + remaining_s * x
        init_y = init_y / largest_sigmas
        
        #setup iteration variables
        x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        #iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            if cls_fn == None:
                et = model(xt, t)
            else:
                et = model(xt, t, classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
            
            if et.size(1) == 6:
                et = et[:, :3]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            # DDNM correction
            # Formula: x0_t_corrected = x0_t + H_pinv * (y_0 - H * x0_t)
            x0_t_flat = x0_t.view(n, -1)
            y_0_flat = y_0.view(n, -1)
            
            h_x0_t = H_funcs.H(x0_t_flat)
            diff = y_0_flat - h_x0_t
            correction = H_funcs.H_pinv(diff)
            
            x0_t_corrected_flat = x0_t_flat + correction
            x0_t_corrected = x0_t_corrected_flat.view_as(x0_t)

            # Compute xt_next from x0_t_corrected and et using DDIM update rule
            pred_dir = et # Direction pointing to x_t

            # Simplified DDIM step (eta=0)
            xt_next = at_next.sqrt() * x0_t_corrected + (1 - at_next).sqrt() * pred_dir

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))


    return xs, x0_preds

def ddnm_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        #iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            if cls_fn == None:
                et = model(xt, t)
            else:
                et = model(xt, t, classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
            
            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            # Correct the prediction using the DDNM null-space projection
            # This is the core of the Coherent-DDRM idea from the paper
            # x0_corrected = x0_t + H_pinv * (y - H * x0_t)
            x0_t_flat = x0_t.view(n, -1)
            y_0_flat = y_0.view(n, -1)
            
            h_x0_t = H_funcs.H(x0_t_flat)
            diff = y_0_flat - h_x0_t
            correction = H_funcs.H_pinv(diff)
            
            x0_t_corrected_flat = x0_t_flat + correction
            x0_t_corrected = x0_t_corrected_flat.view_as(x0_t)

            # Compute the next step xt_next using the corrected x0_t
            # This uses the DDIM update rule
            pred_dir = et # Direction pointing to x_t from the model

            # DDIM step (eta=0 for deterministic sampling)
            xt_next = at_next.sqrt() * x0_t_corrected + (1 - at_next).sqrt() * pred_dir

            x0_preds.append(x0_t)
            xs.append(xt_next)

    return xs, x0_preds