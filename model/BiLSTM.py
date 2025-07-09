import torch
import torch.nn as nn


class BiLSTMLayer(nn.Module):
    def __init__(
        self,
        input_size,
        debug=False,
        hidden_size=512,
        num_layers=1,
        dropout=0.3,
        bidirectional=True,
    ):
        super(BiLSTMLayer, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        self.debug = debug
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

    def forward(self, src_feats):
        packed_outputs, hidden = self.rnn(src_feats)

        rnn_outputs = packed_outputs.data
        if self.bidirectional:
            hidden = self._cat_directions(hidden)

        if isinstance(hidden, tuple):
            hidden = torch.cat(hidden, 0)

        return {"predictions": rnn_outputs, "hidden": hidden}

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


if __name__ == "__main__":
    model = BiLSTMLayer(
        2048,
        debug=False,
        hidden_size=1024,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
    )
    x = torch.randn(180, 32, 2048)
    x_lens = torch.randint(1, 10, (32,))
    output = model(x)
