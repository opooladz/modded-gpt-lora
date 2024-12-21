import os
import torch
from peft import get_peft_model, LoraConfig, TaskType
import wandb
from helper import GPT, GPTConfig, DistributedDataLoader, DAdaptScheduleFreeMuon
import torch.nn.functional as F

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    Simplified version for single GPU training.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, 
                       backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                g = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                
                # Apply orthogonalization
                g = zeropower_backend(g, steps=group['backend_steps'])
                g *= max(1, g.size(0)/g.size(1))**0.5
                
                # Update parameters
                p.data.add_(g, alpha=-lr)

import os
import torch
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self, muon_params, lr=0.002, momentum=0.95, nesterov=True, ns_steps=6,
                 adamw_params=None, adamw_lr=3e-4, adamw_betas=(0.95, 0.95), adamw_eps=1e-8, adamw_wd=0):

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_lr_ratio=adamw_lr/lr, adamw_betas=adamw_betas,
                        adamw_eps=adamw_eps, adamw_wd=adamw_wd)

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]['use_muon'] = True
            else:
                self.state[p]['use_muon'] = False
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]['use_muon'] = False

        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
        else:
            self.world_size = 1
            self.rank = 0

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group['params'] if self.state[p]['use_muon']]
            lr = group['lr']
            momentum = group['momentum']

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(params):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        continue
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    g = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in params:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group['params'] if not self.state[p]['use_muon']]
            lr = group['adamw_lr_ratio'] * group['lr'] # in order for lr schedule to work
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']
            weight_decay = group['adamw_wd']

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['moment1'] = torch.zeros_like(g)
                    state['moment2'] = torch.zeros_like(g)
                state['step'] += 1
                step = state['step']
                buf1 = state['moment1']
                buf2 = state['moment2']
                buf1.lerp_(g, 1-beta1)
                buf2.lerp_(g.square(), 1-beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr/scale)

        return loss

def train_lora():
    # Initialize wandb
    wandb.init(project="gpt2-lora-shakespeare")
    
    # Load base model and add LoRA
    base_model_path = "/mnt/rd3/someonepoe/model_20241221_124308.pt"
    checkpoint = torch.load(base_model_path, weights_only=False)  # Less secure but works with custom classes
    
    # Create config with all required attributes
    config = checkpoint['config']
    if not hasattr(config, 'model_type'):
        config.model_type = "gpt2"
    if not hasattr(config, 'architectures'):
        config.architectures = ["GPT2LMHeadModel"]
    if not hasattr(config, '_name_or_path'):
        config._name_or_path = "gpt2"
    
    # Clean up the state dict by removing "_orig_mod." prefix
    cleaned_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('_orig_mod.'):
            cleaned_state_dict[k[10:]] = v  # Remove first 10 characters ('_orig_mod.')
        else:
            cleaned_state_dict[k] = v
            
    model = GPT(config)
    model.load_state_dict(cleaned_state_dict)
    
    # LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # rank of LoRA
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_q", "c_k", "c_v", "c_proj", "c_fc"]  # Add LoRA to attention and MLP layers
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, peft_config)
    
    # Verify base model is frozen
    for name, param in model.named_parameters():
        if 'lora' not in name:  # if it's not a LoRA parameter
            assert not param.requires_grad, f"Parameter {name} is not frozen!"
        else:  # if it is a LoRA parameter
            assert param.requires_grad, f"LoRA parameter {name} is frozen!"
    
    model.cuda()

    # Data loading using existing DistributedDataLoader
    train_B = 8  # training batch size
    train_T = 1024  # training sequence length
    val_B = 4  # smaller validation batch size
    val_T = 512  # smaller validation sequence length
    
    train_loader = DistributedDataLoader('/mnt/rd/dataset/shakespeare_train_*.bin', train_B, train_T, 0, 1)
    val_loader = DistributedDataLoader('/mnt/rd/dataset/shakespeare_val_*.bin', val_B, val_T, 0, 1)
    
    # Initialize optimizer
    # Separate parameters for Muon and AdamW
    # For LoRA parameters in the transformer blocks
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only look at trainable (LoRA) parameters
            if 'lora' in name and any(layer in name for layer in ['c_q', 'c_k', 'c_v', 'c_proj', 'c_fc']):
                if param.ndim >= 2:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)
            else:
                adamw_params.append(param)
    
    # Initialize optimizer with separate parameter groups
    optimizer = Muon(
        muon_params,
        lr=0.0005,  # Higher learning rate for Muon
        momentum=0.95,
        nesterov=True,
        ns_steps=6,
        adamw_params=adamw_params,
        adamw_lr=3e-5,  # Lower learning rate for AdamW
        adamw_betas=(0.90, 0.95),
        adamw_wd=0.01
    )    
    # Training loop
    model.train()
    
    for step in range(1000000):
        # Get batch
        x, y = train_loader.next_batch()
        
        # Forward pass
        outputs = model(x, y)
        
        # Handle different output types
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        else:
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            if logits.dim() == 3:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            else:
                loss = logits
        
        loss = loss.mean()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Log metrics and validation
        if step % 10 == 0:
            wandb.log({
                'train/loss': loss.item(),
                'step': step
            })
            
            # Validation
            model.eval()
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                val_outputs = model(x_val, y_val)
                
                # Handle different output types for validation (same logic as training)
                if isinstance(val_outputs, dict) and 'loss' in val_outputs:
                    val_loss = val_outputs['loss']
                else:
                    val_logits = val_outputs['logits'] if isinstance(val_outputs, dict) else val_outputs
                    if val_logits.dim() == 3:
                        val_loss = F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), y_val.view(-1))
                    else:
                        val_loss = val_logits
                
                # Ensure validation loss is scalar
                val_loss = val_loss.mean()
                
                wandb.log({
                    'val/loss': val_loss.item(),
                    'step': step
                })
            model.train()
        
        # Save checkpoint periodically
        if step % 10000 == 0:
            model.save_pretrained(f"shakespeare_lora_checkpoint_step_{step}")

    wandb.finish()

if __name__ == "__main__":
    train_lora()
