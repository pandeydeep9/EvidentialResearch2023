import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ep_per_batch=1):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.ep_per_batch = ep_per_batch

        label = np.array(label)
        self.catlocs = []
        for c in range(max(label) + 1):
            # print("C: ", c , np.argwhere(label == c))
            # print("C: ", c , np.argwhere(label == c).reshape(-1))
            self.catlocs.append(np.argwhere(label == c).reshape(-1))

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                classes = np.random.choice(len(self.catlocs), self.n_cls,
                                           replace=False) #n_cls unique clases
                for c in classes:
                    l = np.random.choice(self.catlocs[c], self.n_per,
                                         replace=False) #shot+query random instances from the class
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                # print("7. Episode: ", episode) #for 5w 1s, torch tensor of shape 5*16
                batch.append(episode)
            # print("8. batch: ", batch)
            batch = torch.stack(batch) # bs * n_cls * n_per
            # print("9. the view: ", batch.view(-1))
            yield batch.view(-1)

