import torch 
from torch import nn 
from typing import Tuple, List, Dict, Optional
from pydantic import BaseModel
from nanogpt.model import GPT, GPTConfig
from compress import ArithmeticCoding
from torch.nn import functional as F


class GPTParams(BaseModel):
    n_layer:int=12
    n_head:int=12
    n_embd:int=768
    block_size:int=1024
    bias:Optional[bool]=False
    vocab_size:Optional[int]=2000
    dropout:Optional[float]=0



class LMCompressorBase(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compressor:ArithmeticCoding = ArithmeticCoding()
        self.softmax = nn.Softmax(dim=0)
    def set_params(self, params):
        self.train_params = params
    def train_step(self, X:torch.tensor, Y:torch.tensor) -> Tuple:
        """
        X: should be [BS, T] long
        Y: should be same
        returns logits, loss
        """
        raise NotImplementedError('abstract')
    def inference_step(self, X:torch.tensor, Y:torch.tensor):
        raise NotImplementedError('abstract')
    def init_model(self, params=None):
        raise NotImplementedError()
    def get_step_size(self) -> int:
        return 1
    def configure_optimizer(self) -> Tuple:
        """
        return optimizer, scheduler
        """
        raise NotImplementedError('abstract')
    def compress_step(self, X, Y) -> Tuple[torch.tensor, torch.tensor]:
        """
        returns loss, masked_loss
        """
        logits, loss = self.train_step(X, Y)
        # do compression with logits here
        assert logits.size(0) == 1, f'batchsize should be one but found shape {logits.size()}'
        assert logits.size(1) == X.size(1), f'len should be same as X ({X.size()}) but found shape {logits.size()}'
        probs = self.softmax(logits[0][-1].view(-1))
        self.compressor.encode_token(Y[0][-1].item(), probs)
        return loss



class GPTCompressorBase(LMCompressorBase):
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = self.init_model(params)
        self.set_params(params)
        self.softmax = nn.Softmax(dim=1)
    def train_step(self, X:torch.tensor, Y:torch.tensor):
        return self.model(X, Y)
    def configure_optimizer(self) -> Tuple:
        optimizer = self.model.configure_optimizers(self.train_params.weight_decay, self.train_params.lr, (self.train_params.beta1, self.train_params.beta2), self.train_params.device_type)
        # TODO figure out scheduler
        scheduler = None
        if self.train_params.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.train_params.cycle_steps)
        return optimizer, scheduler
    def get_step_size(self) -> int:
        return self.train_params.window_size - self.train_params.overlap_size
    def compress_step(self, X, Y) -> Tuple[torch.tensor, torch.tensor]:
        """
        returns loss, masked loss
        """
        logits, loss = self.train_step(X, Y)
        # do compression with logits here
        assert logits.size(0) == 1, f'batchsize should be one but found shape {logits.size()}'
        assert logits.size(1) == X.size(1), f'len should be same as X ({X.size()}) but found shape {logits.size()}'
        masked_logits = logits[:, self.train_params.overlap_size:]
        probs = self.softmax(masked_logits.view(-1, masked_logits.size(-1)))
        for i in range(len(probs)):
            self.compressor.encode_token(Y[0][self.train_params.overlap_size + i].item(), probs[i])
        masked_loss = F.cross_entropy(masked_logits.view(-1, logits.size(-1)), Y[:,self.train_params.overlap_size:].view(-1), ignore_index=-1)
        return loss, masked_loss

class GPTSmall(GPTCompressorBase):
    def init_model(self, params):
        params = GPTParams(n_layer=4, n_head=4, n_embd=128, block_size=params.window_size, 
                           bias=False, vocab_size=params.vocab_size, dropout=0)
        model_args = params.dict()
        self.gpt_params = model_args
        gptconf = GPTConfig(**model_args)
        return GPT(gptconf)
 

class LSTMBase(LMCompressorBase): 
    def configure_optimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_params.lr, betas=(self.train_params.beta1, self.train_params.beta2))
        scheduler = None
        if self.train_params.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.train_params.cycle_steps)
        return optimizer, scheduler

    def get_step_size(self) -> int:
        return self.train_params.window_size

   

# Model definition
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, use_ortho:bool=True, use_ghazi_init:bool=False, xavier_init:bool=False, embedding_path:str=None, input_bias:bool=False):
        super(LSTMModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        self.init_lstm(embedding_dim, hidden_dim, n_layers=n_layers, input_bias=input_bias, use_ghazi_init=use_ghazi_init, xavier_init=xavier_init)

    def init_lstm(self, input_dim, hidden_dim, n_layers, input_bias=True, use_ortho=False, use_ghazi_init=False, xavier_init=True, embedding_path=None):
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        # Initialize the forget gate bias to 2
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2   # Forget gate bias indices
                bias.data[start:end].fill_(2.)
                if input_bias:
                    bias.data[: start].fill_(-1.)
        if use_ortho:
            self.init_ortho()
        if use_ghazi_init:
            self.ghazi_lstm_init()
        if xavier_init:
            self.init_xavier()
        if embedding_path:
            self.load_embeddings(embedding_path)

    def load_embeddings(self, embedding_path:str):
        sd = torch.load(embedding_path)
        print('..loaded embeddings')
        self.embedding.load_state_dict(sd)

    def save_embeddings(self, path:str):
        torch.save(self.embedding.state_dict(), path)

    def ghazi_lstm_init(self):
        lstm_layer = self.lstm
        N = self.hidden_dim
        # Get weights associated with different gates
        # Assuming bias is False for simplicity
        w_ii, w_if, w_ig, w_io = lstm_layer.weight_ih_l0.chunk(4, 0)
        u_ii, u_if, u_ig, u_io = lstm_layer.weight_hh_l0.chunk(4, 0)
        # Calculate variances
        var_w_f = torch.var(w_if)
        var_u_f = torch.var(u_if)
        # Second condition from the provided constraints
        # The implementation of this step is a simplification and may need refinement based on your needs
        scale_factor = (1 - N * (var_w_f + var_u_f)) ** (1/3)  # Using cube root as there are 3 gates excluding forget gate
        # Adjust the other weights to satisfy the condition
        w_ii.data *= scale_factor
        w_ig.data *= scale_factor
        w_io.data *= scale_factor
        u_ii.data *= scale_factor
        u_ig.data *= scale_factor
        u_io.data *= scale_factor

        
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_ortho(self):
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if "weight_hh" in name:
                    # Get the number of outputs for shaping the orthonormal matrix
                    num_outputs = param.size(0) // 4  # There are 4 gates in LSTM
                    param.data[:num_outputs] = torch.nn.init.orthogonal_(torch.empty(num_outputs, num_outputs))  # for input gate
                    param.data[num_outputs:2*num_outputs] = torch.nn.init.orthogonal_(torch.empty(num_outputs, num_outputs))  # for forget gate
                    param.data[2*num_outputs:3*num_outputs] = torch.nn.init.orthogonal_(torch.empty(num_outputs, num_outputs))  # for cell gate
                    param.data[3*num_outputs:] = torch.nn.init.orthogonal_(torch.empty(num_outputs, num_outputs))  # for output gate
    
    def init_xavier(self):
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if "weight_hh" in name:
                    print('initing hh')
                    # Get the number of outputs for shaping the orthonormal matrix
                    num_outputs = param.size(0) // 4  # There are 4 gates in LSTM
                    param.data[:num_outputs] = torch.nn.init.xavier_normal_(torch.empty(num_outputs, num_outputs))  # for input gate
                    param.data[num_outputs:2*num_outputs] = torch.nn.init.xavier_normal_(torch.empty(num_outputs, num_outputs))  # for forget gate
                    param.data[2*num_outputs:3*num_outputs] = torch.nn.init.xavier_normal_(torch.empty(num_outputs, num_outputs))  # for cell gate
                    param.data[3*num_outputs:] = torch.nn.init.xavier_normal_(torch.empty(num_outputs, num_outputs))  # for output gate
                if "weight_ih" in name:
                    print('initing ih')
                    # Get the number of outputs for shaping the orthonormal matrix
                    num_outputs = param.size(0) // 4  # There are 4 gates in LSTM
                    num_inputs = param.size(1) #// 4  # There are 4 gates in LSTM
                    param.data[:num_outputs] = torch.nn.init.xavier_normal_(torch.empty(num_outputs, num_inputs))  # for input gate
                    param.data[num_outputs:2*num_outputs] = torch.nn.init.xavier_normal_(torch.empty(num_outputs, num_inputs))  # for forget gate
                    param.data[2*num_outputs:3*num_outputs] = torch.nn.init.xavier_normal_(torch.empty(num_outputs, num_inputs))  # for cell gate
                    param.data[3*num_outputs:] = torch.nn.init.xavier_normal_(torch.empty(num_outputs, num_inputs))  # for output gate
        
    def init_hidden(self, batch_size) -> tuple:
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

class LSTMModelBeefy(LSTMModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, use_ortho:bool=True, use_ghazi_init:bool=False, xavier_init:bool=False, **kwargs):
        super(LSTMModelBeefy, self).__init__(vocab_size,embedding_dim,hidden_dim, n_layers,use_ortho, use_ghazi_init, xavier_init, **kwargs)
        self.input_layers = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                          nn.GELU(),
                                          nn.Linear(hidden_dim, hidden_dim))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                          nn.GELU(),
                                          nn.Linear(hidden_dim, vocab_size))
        self.init_lstm(hidden_dim,hidden_dim,n_layers=n_layers, use_ghazi_init=use_ghazi_init,use_ortho=use_ortho,xavier_init=xavier_init, embedding_path=kwargs.get('embedding_path'), input_bias=kwargs['input_bias'])
       
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        embedded = self.input_layers(embedded)
        output, hidden = self.lstm(embedded, hidden)
        decoded = self.decoder(output)
        return decoded, hidden
 


class MegaLSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size, bias=False)

    def init_hidden(self, batch_size:int) -> Tuple[torch.tensor, torch.tensor]:
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

    def forward(self, x, hidden=None):
        x, hidden = self.lstm_cell(x,hidden)
        return x, hidden

    def train_step(self, x):
        return x 

    def regret_loss(self, states) -> torch.tensor:
        """
        define regret as how much further states would have benefitted if done more comutation
        """
        return



