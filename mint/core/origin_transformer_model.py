import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq,0), tf.float32)
    return seq[:,tf.newaxis,tf.newaxis,:]
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size,size)), -1 , 0)
    return mask 
def get_angles(pos,i,d_model):
    angle_rates = 1/np.power(10000,(2*(i//2))/np.float32(d_model))
    return pos * angle_rates
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:,np.newaxis],np.arange(d_model)[np.newaxis,:], d_model)
    print(angle_rads.shape)

    angle_rads[:,0::2] = np.sin(angle_rads[:,0::2])
    angle_rads[:,1::2] = np.cos(angle_rads[:,1::2])

    pos_encoding = angle_rads[np.newaxis,...]
    return tf.cast(pos_encoding, dtype = tf.float32)
#positional_encoding(10,10)

def scaled_dot_product_attention(q,k,v,mask):
    """
    计算注意力权重。
    q,k,v 必须具有匹配的前置维度。
    k,v 必须具有匹配的导数第二个维度，例如：seq_len_k = seq_len_v.
    虽然mask根据其类型有不同的形状，
    但是mask 必须能进行广播转换以便求和。
    参数：
    q:请求的形状 == （..., seq_len_q, depth）
    k:主键的形状 == （...,seq_len_k,depth）
    v:数值的形状 == （...,seq_len_v,depth_v)
    mask: Float 张量，其形状能转换成
    （...,seq_len_q,seq_len_k）.默认为None.
    返回值：
       输出，注意力权重
    """
    matmul_qk = tf.matmul(q,k,transpose_b = True)#(...,seq_len_q,seq_len_k)
    #https://blog.csdn.net/qq_37430422/article/details/105042303 这里解释了为什么要缩放
    #缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1],tf.float32)
    scaled_dot_product_logits = matmul_qk/tf.math.sqrt(dk)
    #将mask加入到缩放的张量上。
    if mask is not None:
        scaled_dot_product_logits+=(mask*-1e9)
    attention_weights = tf.nn.softmax(scaled_dot_product_logits,axis = -1)
    output = tf.matmul(attention_weights,v)
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
        def __init__(self,d_model,num_heads):
            super(MultiHeadAttention,self).__init__()
            self.num_heads = num_heads
            self.d_model = d_model

            assert d_model%self.num_heads==0
            self.depth = d_model//self.num_heads

            self.wq = tf.keras.layers.Dense(d_model)
            self.wk = tf.keras.layers.Dense(d_model)
            self.wv = tf.keras.layers.Dense(d_model)

            self.dense = tf.keras.layers.Dense(d_model)
        def split_heads(self,x,batch_size):
            x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))
            return tf.transpose(x,perm=[0,2,1,3])

        def call(self, v,k,q,mask):
            batch_size = tf.shape(q)[0]
            q=self.wq(q)
            k=self.wk(k)
            v=self.wv(v)

            q = self.split_heads(q,batch_size)
            k = self.split_heads(k,batch_size)
            v = self.split_heads(v,batch_size)

            scaled_attention , attention_weights = scaled_dot_product_attention(q,k,v,mask)
            scaled_attention = tf.transpose(scaled_attention,perm=[0,2,1,3])

            concat_attention = tf.reshape(scaled_attention,(batch_size,-1,self.d_model))

            output = self.dense(concat_attention)

            return output, attention_weights   
def point_wise_feed_forward_network(d_model,dff):
    return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation = 'relu'),tf.keras.layers.Dense(d_model)])


#encoder layer 
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads,dff,rate=0.1):
        super(EncoderLayer,self).__init__()
        self.mha = MultiHeadAttention(d_model,num_heads)
        self.ffn = point_wise_feed_forward_network(d_model,dff)

        self.layernorm1 = tf.keras.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    def call(self,x,training,mask):
        attn_output, _ = self.mha(x,x,x,mask)
        attn_output = self.dropout1(attn_output,training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout1(ffn_output,training = training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

#decoder layer

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads,dff, rate = 0.1):
        super(DecoderLayer,self).__init__()

        self.mha1 = MultiHeadAttention(d_model,num_heads)
        self.mha2 = MultiHeadAttention(d_model,num_heads)

        self.ffn = point_wise_feed_forward_network(d_model,dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self,x,enc_output,training,look_ahead_mask,padding_mask):
        #attn1,attn_weights_block1 = self.mha1(x,x,x,look_ahead_mask)
        attn1,attn_weights_block1 = self.mha1(x,x,x,padding_mask)
        attn1 = self.dropout1(attn1,training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output,enc_output,out1,padding_mask)
        attn2 = self.dropout2(attn2,training=training)

        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output,training=training)

        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


#Encoder

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding,rate=0.1):
        super(Encoder,self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size,d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
    def call(self,x,training,mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x*=tf.math.sqrt(tf.cast(self.d_model,tf.float32)) #这里将x放大是为了使x相对位置编码更大，从而减少位置编码的影响
        x += self.pos_encoding[:,:seq_len,:]

        x = self.dropout(x,training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x,training,mask)
        return x                                                                                                                                                                                                                                                          
class Decoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,d_model,num_heads,dff, maximum_position_encoding, rate=0.1):
        super(Decoder,self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Dense(d_model);
        self.pos_encoding = positional_encoding(maximum_position_encoding,d_model);
        self.dec_layers = [DecoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    def call(self,x,enc_output,training,look_ahead_mask,padding_mask):
        seq_len = tf.shape(x)[1]
        print("seq_len:",seq_len)
        attention_weights = {}
        x = self.embedding(x) #(batch_size,target_seq_len,d_model)
        print("x.shape:",x.shape)
        #这里放大x,是为了让编码信息相对较小
        x*= tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        x+= self.pos_encoding[:,:seq_len,:]
        x = self.dropout(x,training = training)

        for i in range(self.num_layers):
            x,block1,block2 = self.dec_layers[i](x,enc_output,training,look_ahead_mask,padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self,num_layers,d_model,num_heads,dff,input_vocab_size,target_vocab_size,pe_input,pe_target, rate=0.1):
        super(Transformer,self).__init__()
        self.encoder = Encoder(num_layers,d_model,num_heads,dff,input_vocab_size,pe_input,rate)
        self.decoder = Decoder(num_layers,d_model,num_heads,dff,target_vocab_size,pe_target,rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp,training,enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar,enc_output,training, look_ahead_mask,dec_padding_mask)
        final_output = self.final_layer(dec_output)#(batch_size,tar_seq_len,target_vocab_size)
        return final_output, attention_weights




        
        


if __name__ == "__main__":
    def print_out(q, k, v):
      temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
      print ('Attention weights are:')
      print (temp_attn)
      print ('Output is:')
      print (temp_out)
    np.set_printoptions(suppress=True)
    temp_k = tf.constant([[10,0,0],
                      [0,10,0],
                      [0,0,10],
                      [0,0,10]], dtype=tf.float32)  # (4, 3)
    temp_v = tf.constant([[   1,0],
                      [  10,0],
                      [ 100,5],
                      [1000,6]], dtype=tf.float32)  # (4, 2)
    # 这条 `请求（query）符合第二个`主键（key）`
    # # 因此返回了第二个`数值（value）`。
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)







