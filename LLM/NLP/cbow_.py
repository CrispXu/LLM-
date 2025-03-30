import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm,trange
import numpy as np

#设置上下文词的个数
CONTEXT_SIZE = 2

#原始文本内容
raw_text =""""We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

#构建词汇表
vocab = set(raw_text)
vocab_size = len(vocab)
word2idx = {word:i for i, word in enumerate(vocab)}
idx2word = {i:word for i, word in enumerate(vocab)}

#构建训练数据集
data = []
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    context=(
        [raw_text[i - 2 + j] for j in range(CONTEXT_SIZE)]
        +[raw_text[i + j +1] for j in range(CONTEXT_SIZE)]
    )
    target = raw_text[i]
    data.append((context,target))


def make_context_vector(context,word2idx):
    idxs = [word2idx[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

#定义CBOW模型
class CBOW(nn.Module):
    def __init__(self,vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, 128)
        self.output = nn.Linear(128,vocab_size)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = F.relu(self.proj(embeds))
        out = self.output(out)
        nll_prob = F.log_softmax(out,dim=1)
        return nll_prob

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

model = CBOW(vocab_size,10).to(device)
optimizer = optim.Adam(model.parameters(),lr=0.001)
losses=[]
loss_function = nn.NLLLoss()

for epoch in trange(200):
    total_loss = 0
    for context, target in tqdm(data):
        context_vector = make_context_vector(context, word2idx).to(device)
        target = torch.tensor([word2idx[target]]).to(device)
        model.zero_grad
        train_preidct = model(context_vector)
        loss = loss_function(train_preidct,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)

# 获取词向量表示
context = ['People','create','to','direct']
context_vector = make_context_vector(context, word2idx).to(device)
 
model.eval()
predict =model(context_vector)
max_idx = predict.argmax(1)
predicted_word = idx2word[max_idx.cpu().item()]  # 关键修正
print(predicted_word)
 
print("cBOW embedding'weight=", model.embeddings.weight)
W=model.embeddings.weight.cpu().detach().numpy()#
print(W)
 
#'保存训练后的词向量为npz文件"
np.savez('word2vec实现.npz',file_1=W)
data = np.load('word2vec实现.npz')
print(data.files)

