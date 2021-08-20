

from torch import nn, LongTensor, FloatTensor


class AttnMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask: LongTensor) -> FloatTensor:
        """
        Converts simple input mask into a set of square attention masks to be used
        right before applying the softmax on the attention scores

        Args:
            mask: human-readable LongTensor with values from [0, 1], with dimensions (batch, seqlen)
            e.g.:
                [[1, 1, 1, 1, 0],
                 [1, 1, 1, 0, 0],
                 [1, 0, 0, 0, 0]]

        Returns:
            square_attnmask: FloatTensor with values from [-inf, 0], with dimensions (batch, 1, seqlen, seqlen), which
            should scale to arbitrary number of heads within any given self-attention layer

            tensor([[[0., 0., 0., 0., -inf],
                    [0., 0., 0., 0., -inf],
                    [0., 0., 0., 0., -inf],
                    [0., 0., 0., 0., -inf],
                    [-inf, -inf, -inf, -inf, -inf]],

                    [[0., 0., 0., -inf, -inf],
                    [0., 0., 0., -inf, -inf],
                    [0., 0., 0., -inf, -inf],
                    [-inf, -inf, -inf, -inf, -inf],
                    [-inf, -inf, -inf, -inf, -inf]],

                    [[0., -inf, -inf, -inf, -inf],
                    [-inf, -inf, -inf, -inf, -inf],
                    [-inf, -inf, -inf, -inf, -inf],
                    [-inf, -inf, -inf, -inf, -inf],
                    [-inf, -inf, -inf, -inf, -inf]]])
        """
        (batch, seqlen) = mask.shape
        mask = mask.unsqueeze(0)                                                            # (1, batch, seqlen)

        # Repeat each row tensor {seqlen} times to create a square
        square_mask = mask.repeat_interleave(seqlen, 1)                                     # (1, batch*seqlen, seqlen)

        # expand seqlen into new dimension e.g. [1, 1, 1, 1, 0] becomes stacked vertically 5 times
        square_mask = square_mask.reshape(batch, seqlen, seqlen)                            # (batch, seqlen, seqlen)

        # multiply each square element-wise by it's transpose, which reflects the zeroes to cover the bottom half
        square_mask = square_mask * square_mask.transpose(-1, -2)                           # (batch, seqlen, seqlen) * (batch, seqlen, seqlen) => (batch, seqlen, seqlen)

        # convert mask values from range [0, 1] into [-inf, 0] respectively
        square_mask = square_mask.masked_fill(square_mask==0, value=-1e9)                   # (batch, seqlen, seqlen)
        square_mask = square_mask.masked_fill(square_mask==1, value=0)                      # (batch, seqlen, seqlen)
        return square_mask.unsqueeze(1)                                                     # (batch, 1, seqlen, seqlen)
