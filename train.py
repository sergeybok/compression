import torch, math
import torch.nn as nn
import torch.optim as optim
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm 
from torch.utils.tensorboard import SummaryWriter


#SEED = 125


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
    

class WikiDataset(Dataset):
    def __init__(self, T:int, tokenizer:ByteLevelBPETokenizer, limit:int=-1):
        self.fp = 'enwik9'
        with open(self.fp, 'r') as f:
            self.dataset = f.read()
            if limit > 0:
                self.dataset = self.dataset[:limit]
        self.T = T 
        self.tokenizer = tokenizer 
        self.last_idx = 0

    def first_token(self):
        s = self.dataset[:40]
        tkns = self.tokenizer.encode(s, add_special_tokens=False).ids
        return tkns[:1]

    def reset(self):
        self.last_idx = 0
        
    def __len__(self):
        return math.ceil(len(self.dataset) / self.T)
    
    def __getitem__(self, idx):
        next_idx = self.last_idx + self.T
        if next_idx < len(self.dataset):
            while self.dataset[next_idx] not in [' ', '\n']:
                next_idx += 1
        snippet = self.dataset[ self.last_idx : next_idx + 1]
        tokens = self.tokenizer.encode(snippet, add_special_tokens=False).ids
        tokens = torch.tensor(tokens)
        # print('tokens', tokens)
        string_len = next_idx - self.last_idx
        self.last_idx = next_idx
        return tokens[:-1], tokens[1:], string_len

def update_rank(logits, target, ranks:torch.tensor):
    if ranks is None:
        ranks = torch.tensor([], dtype=torch.int32)
    # assume batchsize = 1 
    logits = logits[0]
    target = target[0]
    outs = []
    for t in range(logits.size(0)):
        # print(logits.size())
        amax = torch.argsort(logits[t], 0)
        # print(amax.size())
        for i,el in enumerate(amax):
            if el == target[t]:
                outs.append(amax.size(0) - i)
    return torch.cat([ranks, torch.tensor(outs, dtype=torch.int32)])


from pydantic import BaseModel
import json
class Params(BaseModel):
    lr:float= 0.0004
    beta1:float=0.5
    beta2:float=0.999
    warmup:int=40_000
    cycle_steps:int=40_000
    use_ortho:bool=False
    use_ghazi_init:bool=False
    use_xavier:bool=False
    beefy:bool=True
    vocab_size:int = 2000
    embedding_dim:int = 200
    hidden_dim:int = 400
    n_layers:int = 1
    chunk_size:int = 600
    use_scheduler:bool=True
    input_bias:bool=True
    # embedding_path:str='embeddings.pt'
    embedding_path:str=None
    seed_start:int=631
    stop_step:int=-1

   

if __name__ == '__main__':


    params = Params()

    # if params.seed_start == 617:
    #     sd = torch.load(f'logs/seed_601/lstm_model.ckpt')
    #     # print('SD.keys=', sd.keys())
    #     torch.save({ 'weight': sd['embedding.weight'] }, params.embedding_path)

    for seed in range(params.seed_start, 11246):
        # @
        # @333 started scheduler 
        # @377 Started warmup for scheduler
        # @411 Set warmup to 50k and ortho every other time
        # @415 made input on beefy embedding => embedding instead of hidden size

        if seed % 2 == 0:
            params.use_xavier = True
        else:
            params.use_xavier = True
        if seed % 3 == 0:
            params.beefy = True
        else:
            params.beefy = True
        params.use_scheduler = False
        # if seed % 5 == 0:
        #     params.use_scheduler = True
        # else:
        #     params.use_scheduler = False
        RANKS = None


        torch.manual_seed(seed)
        name = f'seed_{seed}'
        writer = SummaryWriter(f'logs/{name}')
        with open(f'logs/{name}/meta.json', 'w') as f:
            json.dump({'params': str(params)}, f)
        
        tokenizer = ByteLevelBPETokenizer('enwik9_tokenizer_2000-vocab.json','enwik9_tokenizer_2000-merges.txt')
        if params.beefy:
            model = LSTMModelBeefy(params.vocab_size, params.embedding_dim, params.hidden_dim, params.n_layers, params.use_ortho, params.use_ghazi_init, params.use_xavier, embedding_path=params.embedding_path, input_bias=params.input_bias)
            print('loaded model')
        else:
            model = LSTMModel(params.vocab_size, params.embedding_dim, params.hidden_dim,params.n_layers, params.use_ortho, params.use_ghazi_init, params.use_xavier, embedding_path=params.embedding_path, input_bias=params.input_bias)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=params.lr, betas=(params.beta1, params.beta2))
        model.load_state_dict(torch.load('logs/seed_624/lstm_model.ckpt'))
        optimizer.load_state_dict(torch.load('logs/seed_624/adamw.ckpt'))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, params.cycle_steps)

        # Training loop
        n_epochs = 1
        batch_size = 1
        dataset = WikiDataset(params.chunk_size, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        global_step = 0
        for epoch in range(1, n_epochs+1):
            model.train()
            total_loss = 4
            hidden = model.init_hidden(batch_size)
            pbar = tqdm(total=len(dataloader))
            for batch_idx, (input_data, target_data, _) in enumerate(dataloader):
                if input_data.nelement() == 0:
                    break
                optimizer.zero_grad()
                output, hidden = model(input_data, hidden)
                # Detach the hidden states to prevent backpropagating through the entire sequence
                hidden = (hidden[0].detach(), hidden[1].detach())
                loss = criterion(output.view(-1, params.vocab_size), target_data.view(-1))
                loss.backward()
                optimizer.step()
                if params.use_scheduler and not (epoch == 1 and batch_idx < params.warmup):
                    scheduler.step()
                total_loss = (loss.item() + total_loss) / 2
                if batch_idx % 200 == 0:
                    pbar.set_description(f'E:{epoch} L={round(loss.item(), 4)} S={target_data.size()} R={RANKS.size() if RANKS is not None else 0}')
                    writer.add_scalars('train_loss', { 'loss' : total_loss }, global_step)
                if batch_idx % 25000 == 0:
                    RANKS = update_rank(output, target_data, RANKS)
                    writer.add_histogram(f'tgt_ranks', RANKS, global_step=global_step)
                if batch_idx % 100_000 == 0 and batch_idx > 0:
                    torch.save(model.state_dict(), f'logs/{name}/lstm_model_{batch_idx}.ckpt')
                    # with open(f'logs/{name}/params.txt', 'w') as f:
                    #     json.dump({'params': str(params)}, f)
                if batch_idx == params.stop_step:
                    break
                pbar.update()
                global_step += 1
            print('Saving model')
            torch.save(model.state_dict(), f'logs/{name}/lstm_model.ckpt')
            torch.save(optimizer.state_dict(), f'logs/{name}/adamw.ckpt')
            model.save_embeddings(params.embedding_path)
            dataset.reset()




