import torch
import torch.nn.functional as F
import pdb


def flag(model_forward, perturb_shape, y, step_size, m, optimizer, device, criterion, scalar = False):
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    # import code
    # code.interact(local=locals())

    perturbs = {k: torch.FloatTensor(*x).uniform_(-step_size, step_size).to(device) for k, x in perturb_shape.items()}

    for k, perturb in perturbs.items():
        perturb.requires_grad_()
        out = forward(k, perturb)
        loss = criterion(out, y)
        loss /= m

        for _ in range(m - 1):
            loss.backward()
            perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            #print (perturb.grad)
            perturb.grad[:] = 0

            out = forward(k, perturb)
            loss = criterion(out, y)
            loss /= m

    if scalar is not None:
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()
    else:
        L1 = loss
        loss_train = L1
        loss_train.backward()
        optimizer.step()

    #print(perturbs['A'][0][0:10])

    return loss, out


def flag_biased(model_forward, perturb_shape, y, args, optimizer, device, criterion, training_idx):
    unlabel_idx = list(set(range(perturb_shape[0])) - set(training_idx))

    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-args.step_size, args.step_size).to(device)
    perturb.data[unlabel_idx] *= args.amp
    perturb.requires_grad_()
    out = forward(perturb)
    loss = criterion(out, y)
    loss /= args.m

    for _ in range(args.m - 1):
        loss.backward()

        perturb_data_training = perturb[training_idx].detach() + args.step_size * torch.sign(
            perturb.grad[training_idx].detach())
        perturb.data[training_idx] = perturb_data_training.data

        perturb_data_unlabel = perturb[unlabel_idx].detach() + args.amp * args.step_size * torch.sign(
            perturb.grad[unlabel_idx].detach())
        perturb.data[unlabel_idx] = perturb_data_unlabel.data

        perturb.grad[:] = 0
        out = forward(perturb)
        loss = criterion(out, y)
        loss /= args.m

    loss.backward()
    optimizer.step()

    return loss, out