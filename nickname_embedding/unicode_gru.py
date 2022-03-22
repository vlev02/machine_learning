import torch


class NickNameEncoder(torch.nn.Module):
    def __init__(self, embedding_dims, hidden_size, num_layers, dropout=0.0, pool_type="avg_pool"):
        super().__init__()
        self.bi_gru = torch.nn.GRU(
            input_size=embedding_dims,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
            bidirectional=True)
        self.pool_type = pool_type

    def forward(self, *series):
        t_input, len_li = series
        t_input = t_input.to(torch.float)

        desc_ind = len_li.argsort(descending=True)
        rest_ind = desc_ind.argsort()
        t_input_sorted, len_li_sorted = t_input.index_select(0, desc_ind), len_li.index_select(0, desc_ind)

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            input=t_input_sorted, lengths=len_li_sorted, batch_first=True, enforce_sorted=True)
        packed_bi_output = self.bi_gru(packed_input)

        if self.pool_type == "max_pool":
            # max_pool
            gru_output_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_bi_output[0], padding_value=float("-inf"))
            output_maxpool = gru_output_padded.index_select(1, rest_ind).max(axis=0)[0]
            gru_output = output_maxpool
        elif self.pool_type == "avg_pool":
            # avg_pool
            gru_output_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_bi_output[0], padding_value=.0)
            output_avgpool = gru_output_padded.index_select(1, rest_ind).sum(axis=0) / len_li.reshape(-1, 1)
            gru_output = output_avgpool
        elif self.pool_type == "last_status":
            # last_status
            gru_output_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_bi_output[0], padding_value=.0)
            n_t, n_b, n_h = gru_output_padded.shape
            ind_gather = (len_li - 1).reshape(1, -1, 1).expand(1, n_b, n_h).to(torch.int64)
            output_last = gru_output_padded.index_select(1, rest_ind).gather(0, ind_gather).squeeze()
            gru_output = output_last
        else:
            raise ValueError
        out_norm = torch.nn.functional.normalize(gru_output)
        return out_norm


if __name__ == "__main__":
    encoder = NickNameEncoder(embedding_dims=16, hidden_size=16, num_layers=2, dropout=0.2)
