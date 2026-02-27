import torch
from torch.optim import Optimizer

def find_roots_in_range(serch_range, get_beta,M,lam,q,s,t,granularity,f_roots,device):
    def f(alpha):
            beta=get_beta(alpha)
            term1=M * (beta-1) * t**2
            term2=lam * q * torch.pow(alpha,q) * torch.pow(beta, q-1) * (s * t)**q
            return term1+term2

    r_min,r_max=serch_range[0],serch_range[1]
    probes = torch.linspace(r_min, r_max, granularity,device=device) # split the search_range to probes
    f_value=f(probes)
    sign_check= f_value[:-1]*f_value[1:] < 0
    index=torch.where(sign_check)[0]
    for i in index:
        a1,a2=probes[i],probes[i+1]
        f1=f_value[i]
        target_alpha = (a1+a2)/2
        for _ in range(10):
            f_mid = f(target_alpha)
            if f_mid * f1 < 0:
                a2 = target_alpha
            else:
                a1 = target_alpha
                f1 = f_mid
        target_alpha=(a1+a2)/2
        target_beta=get_beta(target_alpha)
        f_roots.append((target_alpha,target_beta),) # storing result
    return f_roots

def create_custom_optimizer(model, args):
    ab_params=[]

    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            p_a = module.lora_A if isinstance(module.lora_A, torch.nn.Parameter) else getattr(module.lora_A, 'weight', None)
            p_b = module.lora_B if isinstance(module.lora_B, torch.nn.Parameter) else getattr(module.lora_B, 'weight', None)
            
            if p_a is not None and p_b is not None:
                ab_params.append([p_a,p_b])

    optimizer = Custom_Optimizer(
        ab_params,
        q=0.5,
        lr=1e-3
    )
    return optimizer

class Custom_Optimizer(Optimizer):
    def __init__(self, ab_params,lr, M=10, lam=10, q=0.5, granularity=1000):
        params = []
        for i in ab_params:
            params.append({'params': i})
        defaults = dict(M=M, lam=lam, q=q, granularity=granularity,lr=lr)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            M = group['M']     
            lam = group['lam']
            q = group['q']
            gran = group['granularity']
            p_a, p_b = group['params']
            current_device = p_a.device
            if p_a.grad is None or p_b.grad is None:
                continue

            rank = p_b.shape[1]
            num_groups = p_a.shape[0] // rank
            b_rows=p_b.shape[0]// num_groups

            a_tilde_full = p_a.data - (1/M) * p_a.grad.data
            b_tilde_full = p_b.data - (1/M) * p_b.grad.data
            a_tilda_list = a_tilde_full.split(rank, dim=0)
            b_tilda_list = b_tilde_full.split(b_rows, dim=0)
            
            a_list = p_a.data.split(rank, dim=0)
            b_list = p_b.data.split(b_rows, dim=0)

            for g in range(num_groups):
                a=a_tilda_list[g]
                b=b_tilda_list[g]
                for i in range(rank):
                    a_i_vec = a[i, :]
                    b_i_vec = b[:, i]
                    
                    s = torch.norm(a_i_vec, p=2).item()
                    t = torch.norm(b_i_vec, p=2).item()
                    alpha, beta = self.range_solve(M, lam, q, s, t, gran,current_device)
                    print(alpha,beta)
                    a_list[g][i, :] = alpha * a_i_vec
                    b_list[g][:, i] = beta * b_i_vec
        return loss



    def range_solve(self,M, lam, q, s, t,granularity,device):
        def objective(alpha, beta):
            return (M/2) * ((alpha-1)**2 * s**2 + (beta-1)**2 * t**2) + lam * (alpha*beta*s*t)**q
        
        dic={(0,0):(M/2) * (s**2 + t**2),
                (0,1):(M/2) * (s**2),
                (1,0):(M/2) * (t**2),
                (1,1):lam*((s*t)**q)} # find the objective value of boundaries
        
        best_alpha, best_beta = min(dic, key=dic.get) # get the best_alpha, best_beta of min objective value
        min_obj = dic[(best_alpha, best_beta)] # get min objective value

        swapped = False
        if s < t: # check if we are solve alpha or beta
            s, t = t, s
            swapped = True
        
        def get_beta1(alpha): # + quadratic formula
            inner = 1 - 4 * alpha * (1 - alpha) * (s**2 / t**2)
            inner=torch.clamp(inner, min=0)
            return (1 + torch.sqrt(inner)) / 2 
        
        def get_beta2(alpha): # - quadratic formula
            inner = 1 - 4 * alpha * (1 - alpha) * (s**2 / t**2)
            inner=torch.clamp(inner, min=0)
            return (1 - torch.sqrt(inner)) / 2
        
        # get the domain
        bound_inner = torch.tensor(1 - (t / s)**2, device=device)
        bound_low = (1 - torch.sqrt(bound_inner)) / 2
        bound_high = (1 + torch.sqrt(bound_inner)) / 2
        search_ranges = [(1e-9, bound_low), (bound_high, 1 - 1e-9)]

        f_roots=[]
        num_root=0
        for r in search_ranges:
            f_roots=find_roots_in_range(r, get_beta1, M,lam,q,s,t,granularity,f_roots,device) # find root using beta1 formula
            f_roots=find_roots_in_range(r, get_beta2, M,lam,q,s,t,granularity,f_roots,device) # find root using beta1 formula
        if len(f_roots)!=0: # if no any root, stop
            if swapped:
                f_roots = [(y, x) for x, y in f_roots]
            for i in f_roots: # check if existing any root with smaller objective value than the boundary, and get the best root
                target_alpha, target_beta=i[0],i[1]
                current_obj = objective(target_alpha, target_beta)
                if current_obj < min_obj:
                    min_obj = current_obj
                    best_alpha, best_beta = target_alpha, target_beta
        return best_alpha,best_beta