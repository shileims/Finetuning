

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params} - {total_params * 1.e-6:.2f} M params.")
    return total_params

def count_trainable_params(model, verbose=False):
    total_params = sum(p.numel() if p.requires_grad else 0 for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params} - {total_params * 1.e-6:.2f} M params.")
    return total_params