import random

import numpy as np
import pandas as pd
import torch

from unicode_gru import NickNameEncoder


class NameSet(torch.utils.data.Dataset):
    def __init__(self, df, max_len=24):
        self.df = df[df.sc_case >= 0].drop_duplicates(["group_id", "role_name"]).sort_values(
            ["group_id", "role_name"]).reset_index(drop=True).reset_index()
        self.group_cases = self.df.groupby("group_id").index.agg(list).to_dict()
        self.max_len = 24

    def __getitem__(self, idx):
        case0 = self.df.iloc[idx]
        groups = self.group_cases[case0.group_id]
        assert len(groups) > 1
        while True:
            case1_idx = random.choice(groups)
            if case1_idx != idx:
                case1 = self.df.iloc[case1_idx]
                break
        while True:
            case2_idx = random.randint(0, self.df.shape[0] - 1)
            if case2_idx not in groups:
                case2 = self.df.iloc[case2_idx]
                break
        imputs, name_lens, name_li = [], [], []
        for case in (case0, case1, case2):
            name_len = len(case.role_name)
            input_ = np.array([(ord(c) >> np.arange(16)) & 1 for c in
                               (case.role_name[:self.max_len] + "\0" * max(self.max_len - name_len, 0))])
            imputs.append(input_), name_lens.append(name_len), name_li.append(case.role_name)
        return np.array(imputs), np.array(name_lens), name_li

    def __len__(self):
        return self.df.shape[0]


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.margin = margin

    def forward(self, output_0, output_1, output_2):
        loss = (output_0 - output_1).norm(dim=-1).sum() + self.relu(self.margin - (output_0 - output_2).norm(dim=-1)).sum()
        return loss


def encoser_train(encoder, name_data_loader, loss_fn, optimizer):
    encoder.train()
    loss_li = []
    for idx, batch_data in enumerate(name_data_loader):
        optimizer.zero_grad()
        batch_embedding, batch_length, _ = batch_data
        output_0 = encoder(batch_embedding[:, 0], batch_length[:, 0])
        output_1 = encoder(batch_embedding[:, 1], batch_length[:, 1])
        output_2 = encoder(batch_embedding[:, 2], batch_length[:, 2])
        loss = loss_fn(output_0, output_1, output_2)
        loss.backward()
        optimizer.step()
        loss_li.append(loss.item())
        print(f"\r{idx:0>3} -- {loss.item():.3f}", end="")
    print(f"\n average loss: {sum(loss_li) / len(loss_li):.3f}")


if __name__ == "__main__":
    encoder = NickNameEncoder(embedding_dims=16, hidden_size=16, num_layers=2, dropout=0.2)
    df_name_group = None  # /*Warning: data should be replaced!*/ df_name_group: pandas.dataFrame 【group_id, role_name】
    name_set = NameSet(df_name_group)
    loss_fn = ContrastiveLoss()
    optimizer = torch.optim.Adam(encoder.parameters())
    name_data_loader = torch.utils.data.DataLoader(name_set, batch_size=7, shuffle=True)
    n_epoch = 100
    for i_epoch in range(n_epoch):
        print(f"epoch {n_epoch} - {i_epoch + 1}:")
        encoser_train(encoder, name_data_loader, loss_fn, optimizer)

    # visualization
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns

    ds = list(name_set)
    labels = name_set.df.group_id.values
    inputs = [d[0][0] for d in ds]
    lens = [d[1][0] for d in ds]
    names = [d[2][0] for d in ds]
    embeddings = encoder(torch.tensor(inputs), torch.tensor(lens)).detach().numpy()

    pca = PCA()
    pca_result = pca.fit_transform(embeddings)

    df_plot = pd.DataFrame({"pca_1": pca_result[:, 0], "pca_2": pca_result[:, 1], "labels": labels})
    df_plot_sub = df_plot.iloc[:70]
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="pca_1", y="pca_2",
        hue="labels",
        # palette=sns.color_palette("hls", len(df_plot_sub.labels.value_counts())),
        data=df_plot_sub,
        legend="full",
        # alpha=0.3
    )
