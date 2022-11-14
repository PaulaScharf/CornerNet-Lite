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
        max_temp = output[:,:,:,0:width-ind]
        cur_temp = output[:,:,:,0:width-ind]
        next_temp = output[:,:,:,ind:width]
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

    max_val = torch.zeros((batch, channel, height), dtype=torch.float, device=device)
    max_ind = torch.zeros((batch, channel, height), dtype=torch.long, device=device)

    input_temp = input.select(3, width - 1)
    max_val.copy_(input_temp)

    max_ind.fill_(width - 1)

    output_temp = output.select(3, width - 1)
    grad_output_temp = grad_output.select(3, width - 1)
    output_temp.copy_(grad_output_temp)

    un_max_ind = max_ind.unsqueeze(3)
    gt_mask = torch.zeros((batch, channel, height), dtype=torch.bool, device=device)
    max_temp = torch.zeros((batch, channel, height), dtype=torch.float, device=device)
    for ind in range(1,width): # for (int32_t ind = 1; ind < width; ++ind) {
        input_temp = input.select(3, width - ind - 1)
        torch.gt(input_temp, max_val, out=gt_mask)

        torch.masked_select(input_temp, gt_mask, out=max_temp)
        max_val.masked_scatter_(gt_mask, max_temp)
        max_ind.masked_fill_(gt_mask, width - ind - 1)

        un_max_ind = max_ind.unsqueeze(3)
        grad_output_temp = grad_output.select(3, width - ind - 1).unsqueeze(3)
        output.scatter_add_(3, un_max_ind, grad_output_temp)

    return [output]

