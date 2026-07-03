"""
案例：注意力机制的计算规则实现

任务描述：
    已知QKV：
    - Q是查询张量，形状[1, 1, 32]
    - K是索引张量，形状[1, 1, 32]
    - V是内容张量，10个单词，每个单词32个特征，形状[1, 10, 32]

    我们的任务：输入查询张量q，通过注意力机制来计算如下信息：
        1、查询张量q的注意力权重分布：查询张量q和10个单词的相关性（相似度）
        2、查询张量q的结果表示：由一个普通的q升级成一个更强大的q
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        """
        初始化注意力层
        Args:
            query_size: 查询张量的维度
            key_size: 键张量的维度
            value_size1: 值张量的维度1（序列长度/单词个数）
            value_size2: 值张量的维度2（特征维度）
            output_size: 输出维度
        """
        super(MyAttention, self).__init__()

        # 注意力权重分布：计算Q和K的相似度
        # 输入维度：query_size + key_size（Q和K拼接）
        # 输出维度：value_size1（注意力权重的数量，即序列长度）
        self.attn = nn.Linear(query_size + key_size, value_size1)

        # 注意力结果表示：将注意力加权后的结果转换为输出维度
        # 输入维度：query_size + value_size2（Q和加权V拼接）
        # 输出维度：output_size
        self.attn_combine = nn.Linear(query_size + value_size2, output_size)

    def forward(self, Q, K, V):
        """
        前向传播，计算注意力权重和输出
        :param Q: 查询张量，形状: [1, 1, query_size] → [1, 1, 32]
        :param K: 键张量，形状: [1, 1, key_size] → [1, 1, 32]
        :param V: 值张量，形状: [1, value_size1, value_size2] → [1, 10, 32]
        :return: output: 注意力结果表示 [1, 1, output_size] → [1, 1, 32]
                 attn_weights: 注意力权重分布 [1, value_size1] → [1, 10]
        """
        # ========================================
        # 步骤1：计算注意力权重分布
        # ========================================

        # 1.1 拼接Q和K，维度变化: [1, 1, 32] + [1, 1, 32] → [1, 1, 64]
        # Q[0] 去掉batch维度: [1, 32]
        # K[0] 去掉batch维度: [1, 32]
        # 拼接后: [1, 64]
        qk_cat = torch.cat((Q[0], K[0]), dim=-1)
        print(f'qk_cat的形状: {qk_cat.shape}')  # [1, 64]

        # 1.2 通过线性层计算注意力得分，维度变化: [1, 64] → [1, 10]
        # 输入: 64 (Q和K拼接后的维度)
        # 输出: 10 (注意力权重的数量，即单词个数)
        attn_scores = self.attn(qk_cat)
        print(f'attn_scores的形状: {attn_scores.shape}')  # [1, 10]

        # 1.3 用softmax()将得分 → 概率分布: [1, 10] → [1, 10]
        # softmax: 将得分转换为概率，所有概率加起来=1
        # dim=-1: 在最后一个维度上做softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        print(f'attn_weights的形状: {attn_weights.shape}')  # [1, 10]

        # ========================================
        # 步骤2：应用注意力权重到值张量V
        # ========================================

        # 2.1 扩展注意力权重维度，以便匹配V的批次维度
        # [1, 10] → [1, 1, 10]（添加一个维度用于bmm）
        attn_weights_expanded = attn_weights.unsqueeze(0)
        print(f'attn_weights_expanded的形状: {attn_weights_expanded.shape}')  # [1, 1, 10]

        # 2.2 使用bmm()函数执行矩阵乘法
        # 维度变化: [1, 1, 10] × [1, 10, 32] = [1, 1, 32]
        # bmm: 批量矩阵乘法，用于加权求和
        # 每个词的V乘以对应的权重，然后求和
        attn_applied = torch.bmm(attn_weights_expanded, V)
        print(f'attn_applied的形状: {attn_applied.shape}')  # [1, 1, 32]

        # ========================================
        # 步骤3：融合原始查询Q和注意力加权后的V
        # ========================================

        # 3.1 拼接Q和V，维度变化: [1, 1, 32] + [1, 1, 32] → [1, 1, 64]
        # 将原始的Q和融合了上下文信息的attn_applied拼接
        output_cat = torch.cat((Q, attn_applied), dim=-1)
        print(f'output_cat的形状: {output_cat.shape}')  # [1, 1, 64]

        # 3.2 通过线性层降维到输出维度
        # 维度变化: [1, 64] → [1, 32] → [1, 1, 32]
        # 将拼接后的向量转换为最终输出维度
        output = self.attn_combine(output_cat)
        print(f'output的形状: {output.shape}')  # [1, 1, 32]

        # ========================================
        # 步骤4：返回结果
        # ========================================
        # output: [1, 1, 32] - 注意力结果表示（更强大的Q）
        # attn_weights: [1, 10] - 注意力权重分布（每个词的关注度）
        return output, attn_weights


if __name__ == '__main__':
    # ========================================
    # 先验知识：假设qkv的特征属性（特征尺寸/特征数是32）
    # ========================================
    # 有QKV：
    # - Q是查询张量其形状[1, 1, 32]
    # - K是索引张量[1, 1, 32]
    # - V是内容10个单词，每个单词32个特征[1, 10, 32]
    #
    # 我们的任务：输入查询张量q，通过注意力机制来计算如下信息：
    # 1、查询张量q的注意力权重分布：查询张量q（要生成的目标）和source原文（10个单词）相关性 [1, 10]
    # 2、查询张量q的结果表示：由一个普通的q升级成一个更强大q；用q和v做bmm运算

    # ========================================
    # 设置参数
    # ========================================
    query_size = 32      # 查询张量的维度
    key_size = 32        # 键张量的维度
    value_size1 = 10     # 单词个数
    value_size2 = 32     # 每个单词的特征维度
    output_size = 32     # 输出维度

    # ========================================
    # 创建QKV张量
    # ========================================
    Q = torch.randn(1, 1, query_size)         # 查询张量 [1, 1, 32]
    K = torch.randn(1, 1, key_size)           # 键张量 [1, 1, 32]
    V = torch.randn(1, value_size1, value_size2)  # 值张量 [1, 10, 32]

    print(f'Q的形状: {Q.shape}')  # [1, 1, 32]
    print(f'K的形状: {K.shape}')  # [1, 1, 32]
    print(f'V的形状: {V.shape}')  # [1, 10, 32]
    print('=' * 50)

    # ========================================
    # 实例化注意力层
    # ========================================
    myattobj = MyAttention(query_size, key_size, value_size1, value_size2, output_size)

    # ========================================
    # 前向传播
    # ========================================
    output, attn_weights = myattobj(Q, K, V)

    # ========================================
    # 打印结果
    # ========================================
    print('=' * 50)
    print('查询张量q的注意力结果表示(更加强大的q): output-->', output.shape, output)
    print('查询张量q的注意力权重分布attn_weights-->', attn_weights.shape, attn_weights)
    print('注意力机制 End')
