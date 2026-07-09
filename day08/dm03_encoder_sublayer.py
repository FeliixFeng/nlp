"""

    sublayer1
        multi-head attention + add + norm

    sublayer2
        feed forward + add + norm
"""


from dm02_encoder_element import *
from dm01_input import *


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.norm = LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):

        # 子层核心逻辑: 两种常见实现方式
        my_result = self.dropout(self.norm(sublayer(x))) + x
        return my_result

def use_sublayer():
    # 1. 准备输入数据
    x = use_position()

    # 2. 创建子层连接
    sublayer_conn = SublayerConnection(d_model=512)

    # 3. 定义子层函数
    # 多头注意力
    result = sublayer_conn(x, lambda x: MultiHeadAttention(512, 8)(x, x, x))
    # 前馈全连接
    result = sublayer_conn(x, lambda x: FeedForward(512, 2048)(x))

    print(f'子层处理结果:{result.shape}')








if __name__ == '__main__':
    use_sublayer()