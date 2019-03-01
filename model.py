# Read the dataset description
# Read or generate p2h, a dictionary of image name to image id (picture to hash)
# Suppress annoying stderr output when importing keras.
# Determine the size of each image
from keras import backend as K
from keras import regularizers
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121, DenseNet169
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetMobile
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
    Lambda, MaxPooling2D, Reshape, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam


def subblock(x, filter, **kwargs):
    y = BatchNormalization()(x)
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y)  # Reduce the number of features to 'filter'
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)  # no activation # Restore the number of original features
    y = Add()([x, y])  # Add the bypass connection
    y = Activation('relu')(y)
    return y


def build_head(branch_model, activation='sigmoid'):
    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:])
    xb_inp = Input(shape=branch_model.output_shape[1:])
    x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x))(x3)
    x = Concatenate()([x1, x2, x3, x4])
    x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x = Flatten(name='flatten')(x)

    # Weighted sum implemented as a Dense layer.
    x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')
    return head_model


def build_siamese(branch_model, head_model, optim, img_shape=(384, 384, 1)):
    img_a = Input(shape=img_shape)
    img_b = Input(shape=img_shape)
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa, xb])
    model = Model([img_a, img_b], x)
    model.compile(optim, loss='binary_crossentropy', metrics=['acc'])
    return model


def build_model_standard(lr, l2, img_shape=(384, 384, 1)):
    regul = regularizers.l2(l2)
    optim = Adam(lr=lr)
    kwargs = {'padding': 'same', 'kernel_regularizer': regul}

    inp = Input(shape=img_shape)  # 384x384x1
    x = Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 96x96x64
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 48x48x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 48x48x128
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 24x24x128
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 24x24x256
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 12x12x256
    x = BatchNormalization()(x)
    x = Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 12x12x384
    for _ in range(4):
        x = subblock(x, 96, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 6x6x384
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 6x6x512
    for _ in range(4):
        x = subblock(x, 128, **kwargs)

    x = GlobalMaxPooling2D()(x)
    branch_model = Model(inp, x)

    head_model = build_head(branch_model, activation='sigmoid')

    model = build_siamese(branch_model, head_model, optim, img_shape=img_shape)

    return model, branch_model, head_model


pretrained = {
    "densenet121": DenseNet121,
    "densenet169": DenseNet169,
    "resnet50": ResNet50,
    "mobilenet": MobileNet,
    "nasnetmobile": NASNetMobile,
    "xception": Xception,
    "inception": InceptionV3
}


def build_model_pretrained(model_name, lr, img_shape=(384, 384, 3)):
    inp = Input(shape=img_shape)
    base_model = pretrained[model_name](weights='imagenet', include_top=False, input_shape=img_shape, input_tensor=inp)
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    branch_model = Model(inputs=base_model.input, outputs=x)
    optim = Adam(lr=lr)
    head_model = build_head(branch_model, activation='sigmoid')
    model = build_siamese(branch_model, head_model, optim, img_shape=img_shape)
    return model, branch_model, head_model


if __name__ == "__main__":
    model, branch_model, head_model = build_model_pretrained("densenet121", 0.001)
    print(model.summary())
