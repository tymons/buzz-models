from torch import nn

class Discriminator(nn.Module):
    """ Dummy class which implements MLP as discriminator """
    def __init__(self, layers_sizes, input_size):
        """ Constructor for discriminator class """
        assert type(layers_sizes) == list
        assert type(input_size) == int
        
        super(Discriminator, self).__init__()

        self.MLP = nn.Sequential()

        self.MLP.add_module(name=f'FC{0}', module=nn.Linear(input_size, layers_sizes[0]))
        self.MLP.add_module(name=f'A{0}', module=nn.ReLU())
        
        for i, (input_size, output_size) in enumerate(zip(layers_sizes[:-1], layers_sizes[1:]), 1):
            self.MLP.add_module(name=f'FC{i}', module=nn.Linear(input_size, output_size))
            self.MLP.add_module(name=f'A{i}', module=nn.ReLU())

        self.MLP.add_module(name=f'FC{len(layers_sizes) + 1}', module=nn.Linear(layers_sizes[-1], 1))
        self.MLP.add_module(name=f'A{len(layers_sizes) + 1}', module=nn.Sigmoid())

    def forward(self, p, q):
        p_log_probability = self.MLP(p)
        q_log_probability = self.MLP(q)

        return p_log_probability, q_log_probability