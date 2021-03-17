
import tensorflow as tf
import pandas as pd
class ResidualBlock(Layer):
    def __init__(self, units): # units表示的是DNN隐藏层神经元数量
        super(ResidualBlock, self).__init__()
        self.units = units

    def build(self, input_shape):
        print("build")
    def call(self, inputs):
        print("call")

if __name__ == "__main__":
    print(tf.__version__)