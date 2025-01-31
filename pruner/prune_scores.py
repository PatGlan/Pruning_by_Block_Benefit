def taylor1Scorer(params, w_fun=lambda a: -a):
    """
    taylor1Scorer, method 6 from ICLR2017 paper
    <Pruning convolutional neural networks for resource efficient transfer learning>
    """
    if len(params.grad[0].shape) == 4:
        score = (params.grad[0] * params.output).abs().mean(-1).mean(
                                -1).mean(0)
    else:
        score = (params.grad[0] * params.output).abs().mean(1).mean(0)

    return score, 0

def taylor2Scorer(w, grad, pow=2):
    """
    method 22 from CVPR2019 paper, best in their paper
    <Importance estimation for neural network pruning>
    """
    #if len(params.grad[0].shape) == 4:
    #    nunits = params.grad[0].shape[1]
    #else:
    #    nunits = params.grad[0].shape[-1]
    # score = (params.weight * params.weight.grad).data.pow(2).view(nunits, -1).sum(1)
    #score = (params.weight * params.weight.grad).data.pow(2)
    score = (w * grad).data.abs().pow(pow)

    # the two calculation is identical
    # score = (params.weight * params.weight.grad).data.pow(2)
    # equal to calculation belows:
    # if type(params)==list:
    #     score = torch.zeros_like(params[0].grad[0])
    #     N = len(params)
    #     for i in range(N):
    #         score+=(params[i].grad[0]*params[i].output)
    #     score = score.sum(1).sum(0).data.pow(2)
    return score, 0

def taylor3Scorer(params):
    """
    method 23 from CVPR2019 paper, full grad
    <Importance estimation for neural network pruning>
    """
    if len(params.grad[0].shape) == 4:
        full_grad = (params.grad[0] * params.output).sum(-1).sum(-1)
    else:
        full_grad = (params.grad[0] * params.output).sum(1)

    score = full_grad.data.pow(2).sum(0)

    return score, 0

def taylor4Scorer(params, w_fun=lambda a: -a):
    """
    method 1 from 2019 NeuIPS paper
    <Are Sixteen Heads Really Better than One>
    """
    if len(params.grad[0].shape) == 4:
        taylor_im = (params.grad[0] * params.output).sum(-1).abs()
        if params.mask is not None:
            taylor_im = taylor_im[params.mask]
        score = taylor_im.sum(-1).sum(0)
        denom = taylor_im.size(0) * taylor_im.size(2)
    else:
        taylor_im = (params.grad[0] * params.output).abs()
        score = taylor_im.sum(1).sum(0)
        denom = taylor_im.size(0) * taylor_im.size(1)
    return score, denom