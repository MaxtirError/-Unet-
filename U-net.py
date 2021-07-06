from numpy.lib.shape_base import expand_dims
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from datetime import datetime

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

def lrelu(x):
    return tf.maximum(x*0.2, x)

def Loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def InputBlock(x, filters, time, kernel_size = 3, strides = 1, padding = 'same'):
    conv_1 = layers.Conv2D(filters= filters, kernel_size= kernel_size, \
        strides= strides, padding= padding, activation= lrelu, name= 'g_conv' + time + '_1')(x)
    return layers.Conv2D(filters=filters, kernel_size=kernel_size,\
        strides= strides, padding= padding, activation= lrelu, name= 'g_conv' + time + '_2')(conv_1)

def UpsampleAndConcat(x1, x2, filters, exp_time=None, exp=False):
    output_channels = filters * 2
    deconv = layers.Conv2DTranspose(filters= filters, kernel_size= 2, \
                          strides= 2, padding= 'same')(x1)  # 上采样（转置卷积方式）
    output_channels = filters
    if not exp:
        deconv_output = tf.concat([deconv, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])
    if exp:
        cons = tf.fill(tf.shape(deconv), exp_time)
        c = tf.cast(tf.slice(cons, [0, 0, 0, 0], [-1, -1, -1, 1]), dtype= tf.float32)
        deconv_output = tf.concat([deconv, x2, c], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2 + 1])
    return deconv_output

def ContractingPathBlock(x, filters, time, kernel_size = 3, strides = 1, padding = 'same'):
    down_sampling = layers.MaxPool2D((2, 2))(x)
    conv_1 = layers.Conv2D(filters= filters, kernel_size= kernel_size \
        , strides= strides, padding= padding, activation= lrelu, name= 'g_conv' + time + '1')(down_sampling)
    return layers.Conv2D(filters= filters, kernel_size = kernel_size \
        , strides= strides, padding= padding, activation= lrelu, name= 'g_conv' + time + '2')(conv_1)

def ExpansivePathBlock(x, con_feature, filters, e, time, exp = False, kernel_size = 3 \
    , strides = 1, padding = 'same'):
    concat_feature = UpsampleAndConcat(x, con_feature, filters= filters, exp_time= e, exp= exp)
    conv_1 = layers.Conv2D(filters= filters, kernel_size= kernel_size \
        , strides= strides, padding= padding, activation= lrelu, name= 'g_conv' + time + '1')(concat_feature)
    return layers.Conv2D(filters= filters, kernel_size= kernel_size \
        , strides= strides, padding= padding, activation= lrelu, name= 'g_conv' + time + '2')(conv_1)

def UNet(input_shape, e):
    inputs = layers.Input(input_shape)

    input_block = InputBlock(inputs, filters= 32, time= '1')

    conv_1 = ContractingPathBlock(input_block, filters= 64, time= '2')
    conv_2 = ContractingPathBlock(conv_1, filters= 128, time= '3')
    conv_3 = ContractingPathBlock(conv_2, filters= 256, time= '4')
    conv_4 = ContractingPathBlock(conv_3, filters= 512, time= '5')

    exp_4 = ExpansivePathBlock(conv_4, conv_3, filters= 256, e= e, time= '6', exp = True)
    exp_3 = ExpansivePathBlock(exp_4, conv_2, filters= 128, e= e, time= '7')
    exp_2 = ExpansivePathBlock(exp_3, conv_1, filters= 64, e= e, time= '8')
    exp_1 = ExpansivePathBlock(exp_2, input_block, filters= 32, e= e, time = '9')

    outputs = layers.Conv2D(filters= 1, kernel_size= 1, activation=None, name = 'g_conv10')(exp_1)
    return tf.keras.Model(inputs= [inputs], outputs= [outputs])


def return_noise(data, exp_time, ratio= 2, dk = 3, ron = 7):
    width, height = data.shape[0:2]
    img = data * exp_time

    DN = np.random.normal(0, np.sqrt(dk*exp_time/((60*60)*ratio)), (width, height))
    RON = np.random.normal(0, ron, (width, height))
    SN = np.random.poisson(np.abs(img/ratio))

    # 以上是噪声基本格式
    noise_img = (SN + RON + DN)/(exp_time/ratio)
    noise_img = np.where(data == 0.00000000e+00, 0.00000000e+00, noise_img)
    # 使用where防止浮点数0.0被加噪

    return noise_img

def black_level(arr, max_num, level=0.1):
    """Prevent to have an image with more than some percentage of zeroes as input
       level - percentage <0,1>; 0.1/10% default"""
    arr = arr.astype(np.int16)
    src = arr
    arr = list(np.hstack(arr))
    per = arr.count(0)/len(arr)
    if max_num > 10:
        level = 0.3
    if per < level or max_num > 15:
        return True
    else:
        return False

def Expand_Dim(src):
    src = expand_dims(src, axis= 0)
    src = expand_dims(src, axis= -1)
    return src

def generate_data(src, IMG_WIDTH = 256, IMG_HEIGHT = 256):
    H, W = src.shape[0:2]
    zero_level = False
    max_num = 0
    while not zero_level:
        xx = np.random.randint(0, H - IMG_HEIGHT)
        yy = np.random.randint(0, W - IMG_WIDTH)
        arr = src[xx:xx + IMG_HEIGHT, yy:yy + IMG_WIDTH]
        zero_level = black_level(arr, max_num)
        max_num += 1
    out = src[xx:xx + IMG_HEIGHT, yy:yy + IMG_WIDTH]
    out = np.where(out < 0.00000000e+00, 0.00000000e+00  , out)
    return out
def text_write_data(file, l):
    """save parameters into text file"""
    file = open(file, "a+")
    for name in l:
        file.write(str(name) + "\n")
    file.close

Origins = np.load('Origin.npy', allow_pickle = True)
print(Origins.shape)

network_path = None

if network_path is None: # the first time
    model = UNet(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), e= 674)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5), loss = Loss, metrics=['accuracy'])
else:
    model = tf.keras.models.load_model(network_path)
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
tf.keras.backend.set_value(model.optimizer.lr, 1e-4)
batch_size = 1
epochs = 1
def Expand(x):
  return np.expand_dims(np.expand_dims(x, axis= 0), axis= -1)
Loss = []
g_Loss = []
Lass_time = 20
for Time in range(Lass_time, 5000):
    cnt = 0
    for origin in np.random.permutation(Origins):
        out = generate_data(origin)
        out_train = Expand(out)
        for r in range(2, 6):
            img = return_noise(out, 674, ratio= r)
            results = model.fit(Expand(img), out_train, batch_size= batch_size, epochs= epochs, callbacks = [tensorboard_callback])
            g_Loss.append(results.history['loss'][-1])
        
        cnt += 1
        if cnt % 50 == 0:
          print(cnt)
          Loss.append(np.mean(g_Loss))
          g_Loss = []
    if Time % 1 == 0:
        network_path = '/traing'+ '_Time' + str(Time) + '_loss' + str(results.history['loss'][-1]) + '_acu' \
            + str(results.history['accuracy'][-1])
        model.save(network_path + '.h5')
        text_write_data("/GlobalLoss.txt", Loss)
        Loss = []