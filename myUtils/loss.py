import torch.nn as nn
import torch

# original
def PositiveCongruentLossOriginal(new_output, target, new_model, old_logits, old_images, mode='train'):
    # An implementation of Focal Distillation-Logit Matching (FD_LM) from
    # Yan, Sijie, et al. "Positive-congruent training: Towards regression-free model updates."
    # Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
    # Implemented by Yash-yee Logan
    # Parameters:
    # new_output = logits from most recent/current model
    # new_model = model in the current round
    # old_model = model in the previous round
    # target = ground truth labels
    # old_images = images used in the previous round to be passed into both the old and new models to get logits

    CE_loss = nn.CrossEntropyLoss()
    PC_loss = nn.MSELoss()
    lamb = 1

    ce = CE_loss(new_output, target)
    if mode == 'train':
        new_logits = new_model(old_images)
        pc = PC_loss(new_logits, old_logits)
        loss = ce + (lamb * pc)
    else:
        print(new_output.cpu().numpy())
        print(old_logits)
        pc = ((new_output.cpu().numpy() - old_logits)**2).mean()
        loss = ce.cpu().numpy() + (lamb * pc)
    return loss

def PositiveCongruentLoss(new_output, target, preds, new_model, old_logits, old_images, old_preds, mode='train'):
    # An implementation of Focal Distillation-Logit Matching (FD_LM) from
    # Yan, Sijie, et al. "Positive-congruent training: Towards regression-free model updates."
    # Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
    # Implemented by Yash-yee Logan
    # Parameters:
    # new_output = logits from most recent/current model
    # new_model = model in the current round
    # old_model = model in the previous round
    # target = ground truth labels
    # old_images = images used in the previous round to be passed into both the old and new models to get logits

    CE_loss = nn.CrossEntropyLoss()
    PC_loss = nn.MSELoss()
    lamb = 1

    ce = CE_loss(new_output, target)
    if mode == 'train':
        new_logits = new_model(old_images)
        correct = old_preds == target
        alpha = torch.FloatTensor([0]).cuda()
        beta = torch.FloatTensor([1]).cuda()
        pc = alpha + (beta * PC_loss(new_logits[correct], old_logits[correct]))
        loss = ce + (lamb * pc)
        # pc = PC_loss(new_logits, old_logits)
        # print('ce loss: ' + str(ce.item()) + ' pc loss ' + str(pc.item()))
    else:
        # correct = old_preds == target
        # if correct:
        #     pc = ((new_output[correct].cpu().numpy() - old_logits)**2).mean()
        # else:
        #     pc = 0
        # loss = ce.cpu().numpy() + (lamb * pc)
        loss = ce

    return loss

