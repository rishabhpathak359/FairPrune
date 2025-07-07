import numpy as np
import torch
from torch.nn.utils import prune
import copy

#from prune_model import prune_model


def prune_model(model, val_loader, worse_performing_group, criterion, device, p_ratio, b_param):

    print("Performing Pruning!!!!!")

    # this whole procedure may need to be repeated multiple times - it is a hyperparameter how many times
    layer_wise_saliency_odd = {}  # this is the disadvantaged group - need to find which it is after the fine-tuning - using validation
    layer_wise_saliency_even = {}
    # these are only for local debugging - will want to take all batches that are available
    seen_odd = False
    seen_even = False

    assert worse_performing_group is not None
    
    for images, labels, groups in val_loader:
        # move tensors to gpu
        images = images.to(device)
        labels = labels.to(device)

        # odd_group_indices = groups % 2 == 1
        # even_group_indices = groups % 2 == 0

        # IMPORTANT ASSUMPTION: The worse performing group has odd indices
        
        if(type(worse_performing_group) == int):
            if(worse_performing_group == 0):
                odd_group_indices = groups == 0
                even_group_indices = groups == 1
            elif(worse_performing_group == 1):
                odd_group_indices = groups == 1
                even_group_indices = groups == 0
        elif(type(worse_performing_group) == str):
            if(worse_performing_group == 'M'):
                odd_group_indices = np.array(groups) == 'M'
                even_group_indices = np.array(groups) == 'F'
            elif(worse_performing_group == 'F'):
                odd_group_indices = np.array(groups) == 'F'
                even_group_indices = np.array(groups) == 'M'

        # split the images and labels into the groups
        odd_images = images[odd_group_indices]
        odd_labels = labels[odd_group_indices]
        even_images = images[even_group_indices]
        even_labels = labels[even_group_indices]

        if len(odd_images) > 0 and not seen_odd:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # compute the gradient of the scalar function with respect to the input tensor
                    try:
                        grads = torch.autograd.grad(criterion(model(odd_images), odd_labels).mean(), param)
                    except:
                        import pdb; pdb.set_trace()
                    saliency = 0.5 * grads[0]**2 * param**2
                    if name not in layer_wise_saliency_odd:
                        layer_wise_saliency_odd[name] = []
                    layer_wise_saliency_odd[name].append(saliency)
            seen_odd = True

        if len(even_images) > 0 and not seen_even:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # compute the gradient of the scalar function with respect to the input tensor
                    grads = torch.autograd.grad(criterion(model(even_images), even_labels).mean(), param)
                    saliency = 0.5 * grads[0]**2 * param**2
                    if name not in layer_wise_saliency_even:
                        layer_wise_saliency_even[name] = []
                    layer_wise_saliency_even[name].append(saliency)
            seen_even = True

    # now calculate the average saliency for each layer
    layer_wise_saliency_odd = {name: torch.mean(torch.stack(layer_wise_saliency_odd[name]), dim=0) for name in layer_wise_saliency_odd.keys()}
    layer_wise_saliency_even = {name: torch.mean(torch.stack(layer_wise_saliency_even[name]), dim=0) for name in layer_wise_saliency_even.keys()}

    beta_param = b_param  # this will be a hyperparameter to tune
    saliency_differences = {}
    for name in layer_wise_saliency_odd.keys():
        saliency_differences[name] = layer_wise_saliency_odd[name] - beta_param * layer_wise_saliency_even[name]

    # make a list so that we can find the threshold for lowest N% of saliency differences
    all_saliency_differences = []
    for name in saliency_differences.keys():
        all_saliency_differences.append(saliency_differences[name].flatten())

    prune_ratio = p_ratio  # prune 10% of the parameters in this round - we may want to do multiple rounds
    all_saliency_differences = torch.cat(all_saliency_differences)
    threshold = torch.kthvalue(all_saliency_differences, int(prune_ratio * len(all_saliency_differences)))[0]

    # create the mask - we will prune the parameters with saliency difference lower than the threshold
    pruning_mask = {}
    for name in saliency_differences.keys():
        pruning_mask[name] = saliency_differences[name] > threshold

    # print("PRUNING MASK: ")
    # print(pruning_mask)

    # prune the model
    # create deep copy of the model
    pruned_model = copy.deepcopy(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            name_parts = ['pruned_model'] + name.split('.')
            name_first = '.'.join(name_parts[:-1])
            name_last = name_parts[-1]

            # if there is a single number in the name, then replace that part with list
            # example: pruned_model.module.layer1.0.conv1 should be pruned_model.module.layer1[0]conv1
            for e in range(12):
                name_first = name_first.replace('.{}.'.format(e), '[{}].'.format(e))
            for e in range(10, 12):
                name_first = name_first.replace('.{}'.format(e), '[{}]'.format(e))
            for e in range(0, 10):
                name_first = name_first.replace('.{}'.format(e), '[{}]'.format(e))
            module_part = eval(name_first)
            prune.custom_from_mask(module_part, name=name_last, mask=pruning_mask[name])

    return pruned_model

