import torch
import math

def norm(tensors):
    return math.sqrt(sum([torch.sum(tensor ** 2).item() for tensor in tensors]))

@torch.no_grad()
def conjugate_gradient(_hvp, b, maxiter=16, tol=0.01, lam=0.0):
    """
    Minimize 0.5 x^T H^T H x - b^T H x, where H is symmetric
    Args:
        _hvp (function): hessian vector product, only takes a sequence of tensors as input
        b (sequence of tensors): b
        maxiter (int): number of iterations
        lam (float): regularization constant to avoid singularity of hessian. lam can be positive, zero or negative
    (Q = H^T H)
    """
    def hvp(inputs):
        with torch.enable_grad():
            outputs = _hvp(inputs)

        outputs = [xx + lam * yy for xx, yy in zip(outputs, inputs)]

        return outputs

    with torch.enable_grad():
        Hb = hvp(b)

    # zero initialization
    xxs = [hb.new_zeros(hb.size()) for hb in Hb]
    ggs = [- hb.clone().detach() for hb in Hb]
    dds = [- hb.clone().detach() for hb in Hb]

    i = 0

    while True:
        i += 1

        with torch.enable_grad():
            Qdds = hvp(hvp(dds))

        # print(dot(ggs, ggs))
        # print(norm(ggs))

        # if dot(ggs, ggs) < tol:
        if norm(ggs) < tol:
            break

        # one step steepest descent
        alpha = - dot(dds, ggs) / dot(dds, Qdds)
        xxs = [xx + alpha * dd for xx, dd in zip(xxs, dds)]

        # update gradient
        ggs = [gg + alpha * Qdd for gg, Qdd in zip(ggs, Qdds)]

        # compute the next conjugate direction
        beta = dot(ggs, Qdds) / dot(dds, Qdds)
        dds = [gg - beta * dd for gg, dd in zip(ggs, dds)]

        if maxiter is not None and i >= maxiter:
            break

    return xxs

def dot(tensors_one, tensors_two):
    ret = tensors_one[0].new_zeros((1, ), requires_grad=True)

    for t1, t2 in zip(tensors_one, tensors_two):
        print(t1.shape, t2.shape)
        ret = ret + torch.sum(t1 * t2)

    return ret

def autograd(outputs, inputs, create_graph=False):
    """Compute gradient of outputs w.r.t. inputs, assuming outputs is a scalar."""
    inputs = tuple(inputs)
    grads = torch.autograd.grad(outputs, inputs, create_graph=create_graph, allow_unused=True)
    return [xx if xx is not None else yy.new_zeros(yy.size()) for xx, yy in zip(grads, inputs)]

def hww_product(follower_loss, follower, tensors):
    d_follower = autograd(follower_loss(), follower.parameters(), create_graph=True)
    return autograd(dot(d_follower, tensors), follower.parameters())

def hxw_product(follower_loss, leader, follower, tensors):
    d_follower = autograd(follower_loss(), follower.parameters(), create_graph=True)
    return autograd(dot(d_follower, tensors), leader.parameters())

def hxw_inv_hww_dw_product(leader_loss, follower_loss, leader, follower):
    
    # (L_train_ww)^-1 (L_test_w) | x represents poisoner model, w represents poisoned model
    inv_hww_dw = conjugate_gradient(
        # (L_train_ww)^-1
        _hvp=lambda tensors: hww_product(follower_loss, leader, follower, tensors),
        # L_test_w
        b=autograd(leader_loss, follower.parameters()))
    
    # L_train_wx ((L_train_ww)^-1 (L_test_w))
    hxw_inv_hww_dw = hxw_product(
        follower_loss,
        leader,
        follower,
        inv_hww_dw
    )

    return hxw_inv_hww_dw
