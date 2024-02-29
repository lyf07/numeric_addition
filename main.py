import model
import dataset
import torch
import numpy as np
import time

ndigit = 2

train_dataset = dataset.AdditionDataSet(ndigit, 'train')
test_dataset = dataset.AdditionDataSet(ndigit, 'test')

config = model.GPTConfig(train_dataset.vocab_size, train_dataset.block_size, embed_dim=512, n_heads=4, blocks=2, expansion_factor=4)
model = model.GPT(config)




from Train import Trainer, TrainerConfig

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=50, batch_size=512, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=1024, final_tokens=50*len(train_dataset)*(ndigit+1),
                      num_workers=0)
trainer = Trainer(model, train_dataset, test_dataset, tconf)
# for x, y in enumerate(trainer.train_dataset):
    # print(f'In main, x = {x}')
    # print(f'In main, y = {y}')
trainer.train()

# now let's give the trained model an addition exam
from torch.utils.data.dataloader import DataLoader
from utils import sample

def give_exam(dataset, batch_size=32, max_batches=-1):
    results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        d1d2 = x[:, :ndigit*2]
        d1d2d3 = sample(model, d1d2, ndigit+1)
        d3 = d1d2d3[:, -(ndigit+1):]
        # d3 = d1d2d3[:, :(ndigit+1)]
        # print(f'd3.shape={d3.shape}')
        # d3 = torch.flip(d3, dims=[1])
        # print(f'd3={d3}')
        factors = torch.tensor([[10**i for i in range(ndigit+1)][::-1]]).to(trainer.device)
        # decode the integers from individual digits
        d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(1)
        d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(1)
        d3i_pred = (d3 * factors).sum(1)
        d3i_gt = d1i + d2i
        correct = (d3i_pred == d3i_gt).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line, lol
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            judge = 'YEP!!!' if correct[i] else 'NOPE'
            if not correct[i]:
                print("GPT claims that %03d + %03d = %03d (gt is %03d; %s)" 
                      % (d1i[i], d2i[i], d3i_pred[i], d3i_gt[i], judge))
        
        if max_batches >= 0 and b+1 >= max_batches:
            break

    print("final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))


give_exam(test_dataset, batch_size=1024, max_batches=10)

time.sleep(10)
ndigit = 1

test_dataset = dataset.AdditionDataSet(ndigit, 'test')
give_exam(test_dataset, batch_size=1024, max_batches=10)
