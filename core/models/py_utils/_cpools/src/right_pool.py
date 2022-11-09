import numpy as np
import torch

def forward(input):    # Initialize output
    output = torch.zeros_like(input)

    # Get width
    width = input.size(3)

    output.copy_(input)

    i = 0
    ind = 2**i

    while ind<width: 
        max_temp = output[:,:,:,ind:width]
        # print(max_temp.size())
        cur_temp = output[:,:,:,ind:width]
        next_temp = output[:,:,:,0:width-ind]
        torch.max(max_temp, cur_temp, out = next_temp)
        i += 1
        ind = 2**i


    return [output]



def backward(input, grad_output):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output = torch.zeros_like(input)

    batch = input.size(0)
    channel = input.size(1)
    height = input.size(2)
    width = input.size(3)

    max_val = torch.zeros((batch, channel, height), torch.from_numpy(input).float().to(device))
    max_ind = torch.zeros((batch, channel, height), torch.from_numpy(input).long().to(device))

    input_temp = input.select(3, 0)
    max_val.copy_(input_temp)

    max_ind.fill_(0)

    output_temp = output.select(3, 0)
    grad_output_temp = grad_output.select(3, 0)
    output_temp.copy_(grad_output_temp)

    un_max_ind = max_ind.unsqueeze(2)
    gt_mask = torch.zeros((batch, channel, height), torch.from_numpy(input).float().to(device))
    max_temp = torch.zeros((batch, channel, height), torch.from_numpy(input).long().to(device))
    for ind in range(width): # for (int32_t ind = 0; ind < width - 1; ++ind)
        input_temp = input.select(3, ind + 1)
        np.gt_out(gt_mask, input_temp, max_val) # does this exist in numpy

        np.masked_select_out(max_temp, input_temp, gt_mask) # does this exist in numpy
        max_val.masked_scatter_(gt_mask, max_temp)
        max_ind.masked_fill_(gt_mask, ind + 1)

        grad_output_temp = grad_output.select(3, ind + 1).unsqueeze(3)
        output.scatter_add_(3, un_max_ind, grad_output_temp)

    return output

