import torch
CUDA = torch.cuda.is_available()  # checking cuda availability
cuda = 0
if CUDA:
    torch.cuda.set_device(cuda)
    print("Choose cuda:{}".format(cuda))