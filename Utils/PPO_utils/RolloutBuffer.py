import torch
import numpy as np

'''用來存一個 trajectory
    作法 : 每一個資料各開一個 list 依照時間步進行儲存
'''

class RolloutBuffer:


    def __init__(self, device):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        # 在我的場景中沒有 done，因此 dones 中永遠都會是 False。因此在 A_{trajectory length} 應該要傳入 V_{trajectory_length} 而不是不考慮
        self.dones = []  
        # 算 GAE 時會用到
        self.values = []
        self.device = device

    
    # 儲存資料
    # state : np.array with shape (state_dim)
    # action : torch.tensor with shape (action_dim)
    # log_prob : torch.tensor with shape () (already detach())
    # reward : float
    # done : float 
    # value : state value of next state, float (by .item())
    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    
    # 更新完後要清除才能存下一個 trajectory
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []  
        self.values = []

    
    def len(self):
        return len(self.states)

    
    # {A}_t^{GAE} = \delta_t + (\gamma \lambda) {A}_{t+1}^{GAE}
    # A_t = \delta_t + (\gamma \lambda) A_{t+1}
    # \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
    # 一定要 reversed 計算 (從 T 往前算每一步的 GAE)，因為 A^{GAE}_{t+1} 要考慮到後面所有時間步
    # 若從前面開始算則得不到後面所有時間步的資訊
    # last_value : float (by item())，假設 trajectory_length = T，這是 V(s_{T+1})
    # output : returns, advantages, shape (trajectory_length)
    def compute_GAE(self, last_value, gamma= 0.99, gae_lambda= 0.95):
        # 轉為 tensor 並把 shape 轉成 (trajectory_length)
        values = torch.tensor(self.values + [last_value], dtype= torch.float32, device= self.device).flatten()
        rewards = torch.tensor(self.rewards, dtype= torch.float32, device= self.device).flatten()
        dones = torch.tensor(self.dones, dtype= torch.float32, device= self.device).flatten()
        returns = []
        advantages = []
        gae = 0
        
        # compute return & GAE of every timesteps (reversely)
        # {A}_t^{GAE} = \delta_t + (\gamma \lambda) {A}_{t+1}^{GAE}
        for step in reversed(range(len(self.rewards))):
            # TD-Error of current timestep t : \delta_t = r(t)+\gamma*V(t+1) - V(t)
            delta = rewards[step] + gamma * values[step+1] * (1 - dones[step]) - values[step]
            # GAE of current timestep t : {A}_t^{GAE} = \delta_t + (\gamma \lambda) {A}_{t+1}^{GAE}
            gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)  # 插入在最前面
            # return = Advantage + value (A = Q(這邊的 return) - V)
            returns.insert(0, gae + values[step])
        
        # 為了讓數值穩定，對 Advantage 做 normalization
        advantages = torch.tensor(advantages, dtype= torch.float32, device= self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return torch.tensor(returns, dtype= torch.float32, device= self.device), advantages

    
    '''因為一次只能利用一個 trajectory 的資料進行更新，因此使用此函式是為了 :
        1. 打破時間相關性 : 若是照著時間一個一個更新，容易導致偏差 (可能會有災難性遺忘)
        2. 將資料以 batch 的形式抽出，平行化更新
       另外，此函式的結尾是一個 yield 而非一個 return，這代表我們的函式是一個 generator 而非一個單純的 function。
        在後面的使用中，我們會這樣 : 
        for batch in gen:
            取出一筆資料 (第一個 batch) 後進行更新。此時這個 get_batch 會停在 yield 那一行
            (到下一個迴圈時) 從上一個迴圈的 yield 那一行開始繼續往下跑 (取第二個 batch)
        由此可以看出 yield 很省記憶體，不用每次都重跑一遍函式，從頭到尾都用同一個函式
    '''
    # returns : torch.tensor with shape (trajectory_length), 此為 Critic 產出的 value 的 Target
    # advantages : torch.tensor with shape (trajectory_length)
    # 回傳一個 tuple，裡面包含五個 tensors
    #     states[idx] : torch.tensor with shape (batch_size, state_dim)
    #     actions[idx] : torch.tensor with shape (batch_size, action_dim)
    #     old_log_probs[idx] : torch.tensor with shape (batch_size)
    #     returns[idx] : torch.tensor with shape (batch_size)
    #     advantages[idx] : torch.tensor with shape (batch_size)
    def get_batch(self, returns, advantages, batch_size):
        # states 為一個裝滿 np.array 的 list，若要直接轉為 tensor 會超慢，先一律轉成 numpy 再轉為 tensor 是最佳的作法，並非多此一舉
        states = torch.tensor(np.array(self.states), dtype= torch.float32, device= self.device)  # shape (trajectory_length, state_dim)
        actions = torch.tensor(np.array(self.actions), dtype= torch.float32, device= self.device)  # shape (trajectory_length, action_dim)
        old_log_probs = torch.tensor(self.log_probs).to(device= self.device)  # shape (trajectory_length)
        # 把一個 trajectory 的資料存成一個 dataset
        dataset_size = len(states)
        # dataset 中各資料的 indices
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)  # 打亂資料的順序
        # 一次吐出一個 batch 的資料
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            idx = indices[start : end]
            yield (states[idx], actions[idx], old_log_probs[idx], returns[idx], advantages[idx])
