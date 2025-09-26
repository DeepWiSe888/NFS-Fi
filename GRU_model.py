
import torch
import torch.nn as nn
import numpy as np


class GRUClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size = 64, num_layers = 1):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.InstanceNorm1d(128),
            nn.Dropout(0.1),
            # nn.BatchNorm1d(128),
            nn.Linear(128, self.hidden_size),
            nn.ReLU(),
            nn.InstanceNorm1d(self.hidden_size),
            nn.Dropout(0.1),
            # nn.BatchNorm1d(self.hidden_size),
        )
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.1)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.InstanceNorm1d(self.hidden_size//2),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size//2, self.hidden_size//2),
        )

        self.classifier = nn.Linear(self.hidden_size//2, num_classes)


    def forward(self, x, seq_len_vec):
        # print(x.size())
        x = self.mlp(x)  # size: (batch_size, seq_len, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # initial state for GRU
        out_gru, _ = self.gru(x, h0)  # size: (batch_size, seq_len, hidden_size)
        batch_indices = torch.arange(x.size(0)).to(x.device) # x.size(0) is the batch size, [0,1,..., batch_size-1]
        out_end = out_gru[batch_indices, seq_len_vec, :]   # size: (batch_size, 1, hidden_size), 1 is the last of seq
        out_feat = self.fc(out_end)
        out = self.classifier(out_feat)
        return out, out_end

    def get_weight(self):
        return self.classifier.weight

    # this is used for testing the model
    def forward_test(self, x, seq_len_vec):
        print('{}: {}'.format('x size: input', x.data.size()))
        x = self.mlp(x)  # size: (batch_size, seq_len, hidden_size)
        print('{}: {}'.format('x size: after mlp', x.data.size()))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # initial state for GRU
        print('{}: {}'.format('h0 size', h0.data.size()))
        out_gru, _ = self.gru(x, h0)  # size: (batch_size, seq_len, hidden_size)
        print('{}: {}'.format('out size: after gru', out_gru.data.size()))
        batch_indices = torch.arange(x.size(0)).to(x.device) # x.size(0) is the batch size, [0,1,..., batch_size-1]
        print('{}: {}'.format('batch_indices', batch_indices.data.size()))
        out_end = out_gru[batch_indices, seq_len_vec, :]   # size: (batch_size, 1, hidden_size), 1 is the last of seq
        print('{}: {}'.format('out size: after split', out_end.data.size()))
        out_feat = self.fc(out_end)
        print('{}: {}'.format('out size: after fc', out_feat.data.size()))
        out = self.classifier(out_feat)
        print('{}: {}'.format('out size: after classifier', out.data.size()))

        print('{}: {}'.format('length of effective data', seq_len_vec))
        print('{}: {}'.format('length of batch_indices', batch_indices))
        return out, out_feat



if __name__ == '__main__':
    import random
    from kmeans_pytorch import kmeans
    seq_len_vec = 100
    DATA_VEC_LEN = 370
    NUM_CLASSES = 10
    batch_size = 64
    time_num = 240
    label = [random.randint(0, NUM_CLASSES-1) for _ in range(batch_size)]
    gru_model = GRUClassifier(input_size=DATA_VEC_LEN, num_classes=NUM_CLASSES)
    print(gru_model)


    sequences = torch.Tensor(batch_size,time_num,DATA_VEC_LEN)  # batch_size, number of samples (time), dim of sample (370)
    indeces = np.random.randint(10, time_num-1, size=(batch_size,))   # ndarray: (batch_size,:)
    out, out_feat = gru_model.forward_test(sequences,indeces)
    # feat_means = out_feat.detach().numpy()
    last_weight = gru_model.get_weight()

    cluster_ids_x, cluster_centers = kmeans(
        X=out_feat, num_clusters=10, distance='euclidean')
    print(cluster_ids_x)
    print('Successful')