import torch
import torch.nn as nn

class criterion_mse(nn.Module):
    def __init__(self):
        super(criterion_mse, self).__init__()
        self.MSELoss = torch.nn.MSELoss(reduction ='none')
        lambda_ = 1.5
        weight_body = 1/(17+6*lambda_)
        weight_feet = lambda_/(17+6*lambda_)
        tmp1 = [weight_body for i in range(17)]
        tmp2 = [weight_feet for i in range(6)]
        self.weights = torch.tensor(tmp1+tmp2).unsqueeze(0)

    def forward(self, input,target):
        """Computing modified MSE loss in paper.
           Args:
               target: a tensor of shape [C, B, H, W] ground truth.
               input: a tensor of shape [C, B, H, W]. Corresponds to the raw output.
               Here, H, W represent heat map size
           Returns:
               proposed loss in paper
           """
        output = self.MSELoss(input, target)
        output = torch.mean(output,dim=(2,3))
        w = self.weights.repeat(output.size()[0],1)
        output = torch.mul(output,w)
        return torch.mean(output)

if __name__ == '__main__':
    # Here is an example of usage
    input = torch.randn(20, 23, 80, 64, requires_grad=True)
    target = torch.randn(20, 23, 80, 64)
    criterion = criterion_mse()
    loss = criterion(input,target)
    print(loss)
