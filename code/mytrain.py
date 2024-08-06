# !/usr/bin/env python
# -*-coding:utf-8 -*-

import torch

def train_emtsr_itemfilter(args, generator_trainloader, etgsr_trainloader, Recmodel, opt, device):
    Recmodel.train()
    batch_loss_avg, etgsr_loss_avg, u2i_loss_avg, l2_loss_avg, generator_loss_avg, i2i_loss_avg, reg_loss_avg =\
        0., 0., 0., 0., 0., 0., 0.
    generator_trainloader = iter(generator_trainloader)
    for batch_i, traindata in enumerate(etgsr_trainloader):
        item1, item2, item_neg = next(generator_trainloader)
        item1 = item1.to(device)
        item2 = item2.to(device)
        item_neg = item_neg.to(device)

        i2iloss = Recmodel.i2i(item1, item2, item_neg)
        generator_loss = i2iloss
        i2i_loss_avg = i2i_loss_avg + i2iloss
        generator_loss_avg = generator_loss_avg + generator_loss

        weight_for_local_update = list(Recmodel.generator.encoder.state_dict().values())
        grad = torch.autograd.grad(generator_loss, Recmodel.generator.encoder.parameters(), create_graph=True, allow_unused=True)
        fast_weights = []
        for i, weight in enumerate(weight_for_local_update):
            fast_weights.append(weight - args.g_lr * grad[i])

        batch_user, batch_pos, batch_neg = traindata
        batch_user = batch_user.to(device)
        batch_pos = batch_pos.to(device)
        batch_neg = batch_neg.to(device)
        u2iloss, l2loss = Recmodel(batch_user, batch_pos, batch_neg, fast_weights)

        etgsr_loss = u2iloss+args.l2decay*l2loss
        u2i_loss_avg = u2i_loss_avg+u2iloss
        l2_loss_avg = l2_loss_avg+l2loss
        etgsr_loss_avg = etgsr_loss_avg+etgsr_loss

        batch_loss = etgsr_loss + args.beta * generator_loss
        batch_loss_avg = batch_loss_avg+batch_loss

        opt.zero_grad()
        batch_loss.backward()
        opt.step()

    batch_loss_avg = batch_loss_avg/len(etgsr_trainloader)
    etgsr_loss_avg = etgsr_loss_avg/len(etgsr_trainloader)
    u2i_loss_avg = u2i_loss_avg/len(etgsr_trainloader)
    l2_loss_avg = l2_loss_avg/len(etgsr_trainloader)
    generator_loss_avg = generator_loss_avg/len(etgsr_trainloader)
    i2i_loss_avg = i2i_loss_avg/len(etgsr_trainloader)
    reg_loss_avg = reg_loss_avg/len(etgsr_trainloader)
    return batch_loss_avg, etgsr_loss_avg, u2i_loss_avg, l2_loss_avg, generator_loss_avg, i2i_loss_avg, reg_loss_avg

