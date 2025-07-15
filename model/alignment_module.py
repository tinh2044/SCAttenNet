import torch
import torch.nn as nn


class AlignmentModule(nn.Module):
    def __init__(
        self,
        cls_num,
        input_size,
        hidden_size,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm_hidden_size = int(hidden_size / self.num_directions)
        self.dropout = dropout

        # BiLSTM layer
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        self.gloss_layer = nn.Linear(self.hidden_size, cls_num)

    def _cat_directions(self, hidden):
        """If the encoder is bidirectional, do the following transformation.
        Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
        -----------------------------------------------------------
        In: (num_layers * num_directions, batch_size, hidden_size)
        (ex: num_layers=2, num_directions=2)

        layer 1: forward__hidden(1)
        layer 1: backward_hidden(1)
        layer 2: forward__hidden(2)
        layer 2: backward_hidden(2)

        -----------------------------------------------------------
        Out: (num_layers, batch_size, hidden_size * num_directions)

        layer 1: forward__hidden(1) backward_hidden(1)
        layer 2: forward__hidden(2) backward_hidden(2)
        """

        def _cat(h):
            return torch.cat([h[0 : h.size(0) : 2], h[1 : h.size(0) : 2]], 2)

        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)

        return hidden

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.permute(1, 0, 2)

        logits = self.gloss_layer(x)

        return logits


if __name__ == "__main__":
    model = AlignmentModule(cls_num=100, input_size=2048, hidden_size=1024)
    x = torch.randn(180, 32, 2048)
    logits = model(x)
    print(f"AlignmentModule output shape: {logits.shape}")
