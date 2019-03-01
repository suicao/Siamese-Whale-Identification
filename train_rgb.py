import argparse
import os
import pickle
import platform
import random
import sys
# from lap import lapjv
from lapjv import lapjv
from math import sqrt
from os.path import isfile

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as pil_image
from imagehash import phash
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence
from pandas import read_csv
from tqdm import tqdm

from model import build_model_pretrained

parser = argparse.ArgumentParser()
parser.add_argument("--train_df", default="/media/khoi/Data/whale/train.csv", type=str)
parser.add_argument("--sub_df", default="/media/khoi/Data/whale/sample_submission.csv", type=str)
parser.add_argument("--train_dir", default='/media/khoi/Data/whale/train/', type=str)
parser.add_argument("--test_dir", default='/media/khoi/Data/whale/test/', type=str)
parser.add_argument("--p2h", default='./metadata/p2h.pickle', type=str)
parser.add_argument("--p2size", default='./metadata/p2size.pickle',
                    type=str)
parser.add_argument("--bb_df", default="./metadata/bounding_boxes.csv", type=str)
parser.add_argument("--model_arch", default='densenet121', type=str)
parser.add_argument("--model_path", default='./models/densenet121_rgb.model', type=str)
parser.add_argument("--img_shape", default='384,384,3', type=str)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lapjv_threads", default=4, type=int)
parser.add_argument("--tta", default=0, type=int)
parser.add_argument("--predict_only", default=0, type=int)
args = parser.parse_args()

if not os.path.exists("./models"):
    os.mkdir("./models")

img_shape = tuple([int(val) for val in args.img_shape.split(',')])  # The image shape used by the model
anisotropy = 2.15  # The horizontal compression ratio
crop_margin = 0.1  # The margin added around the bounding box to compensate for bounding box inaccuracy

tagged = dict([(p, w) for _, p, w in read_csv(args.train_df).to_records()])
submit = [p for _, p, _ in read_csv(args.sub_df).to_records()]
join = list(tagged.keys()) + submit


def expand_path(p):
    if isfile(args.train_dir + p):
        return args.train_dir + p
    if isfile(args.test_dir + p):
        return args.test_dir + p
    return p


if isfile(args.p2size):
    with open(args.p2size, 'rb') as f:
        p2size = pickle.load(f)
else:
    p2size = {}
    for p in tqdm(join):
        size = pil_image.open(expand_path(p)).size
        p2size[p] = size
    with open(args.p2size, 'wb') as f:
        pickle.dump(p2size, f)


# Two phash values are considered duplicate if, for all associated image pairs:
# 1) They have the same mode and size;
# 2) After normalizing the pixel to zero mean and variance 1.0, the mean square error does not exceed 0.1
def match(h1, h2):
    for p1 in h2ps[h1]:
        for p2 in h2ps[h2]:
            i1 = pil_image.open(expand_path(p1))
            i2 = pil_image.open(expand_path(p2))
            if i1.mode != i2.mode or i1.size != i2.size: return False
            a1 = np.array(i1)
            a1 = a1 - a1.mean()
            a1 = a1 / sqrt((a1 ** 2).mean())
            a2 = np.array(i2)
            a2 = a2 - a2.mean()
            a2 = a2 / sqrt((a2 ** 2).mean())
            a = ((a1 - a2) ** 2).mean()
            if a > 0.1: return False
    return True


if isfile(args.p2h):
    with open(args.p2h, 'rb') as f:
        p2h = pickle.load(f)
else:
    # Compute phash for each image in the training and test set.
    p2h = {}
    for p in tqdm(join):
        img = pil_image.open(expand_path(p))
        h = phash(img)
        p2h[p] = h

    # Find all images associated with a given phash value.
    h2ps = {}

    for p, h in p2h.items():
        if h not in h2ps: h2ps[h] = []
        if p not in h2ps[h]: h2ps[h].append(p)

    # Find all distinct phash values
    hs = list(h2ps.keys())

    # If the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
    h2h = {}

    for i, h1 in enumerate(tqdm(hs)):
        for h2 in hs[:i]:
            if h1 - h2 <= 6 and match(h1, h2):
                s1 = str(h1)
                s2 = str(h2)
                if s1 < s2: s1, s2 = s2, s1
                h2h[s1] = s2

    # Group together images with equivalent phash, and replace by string format of phash (faster and more readable)
    for p, h in p2h.items():
        h = str(h)
        if h in h2h: h = h2h[h]
        p2h[p] = h
    with open(args.p2h, 'wb') as f:
        pickle.dump(p2h, f)
# For each image id, determine the list of pictures
h2ps = {}
for p, h in p2h.items():
    if h not in h2ps: h2ps[h] = []
    if p not in h2ps[h]: h2ps[h].append(p)


# Show an example of a duplicate image (from training of test set)


def show_whale(imgs, per_row=2):
    n = len(imgs)
    rows = (n + per_row - 1) // per_row
    cols = min(per_row, n)
    fig, axes = plt.subplots(rows, cols, figsize=(24 // per_row * cols, 24 // per_row * rows))
    for ax in axes.flatten(): ax.axis('off')
    for i, (img, ax) in enumerate(zip(imgs, axes.flatten())): ax.imshow(img.convert('RGB'))


# For each images id, select the prefered image
def prefer(ps):
    if len(ps) == 1: return ps[0]
    best_p = ps[0]
    best_s = p2size[best_p]
    for i in range(1, len(ps)):
        p = ps[i]
        s = p2size[p]
        if s[0] * s[1] > best_s[0] * best_s[1]:  # Select the image with highest resolution
            best_p = p
            best_s = s
    return best_p


h2p = {}
for h, ps in h2ps.items():
    h2p[h] = prefer(ps)
len(h2p), list(h2p.items())[:5]


def read_raw_image(p):
    img = pil_image.open(expand_path(p))
    return img


# Read the bounding box data from the bounding box kernel (see reference above)
p2bb = pd.read_csv(args.bb_df).set_index("Image")

old_stderr = sys.stderr
sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul', 'w')

sys.stderr = old_stderr

datagen = ImageDataGenerator()


def read_cropped_image(p, augment):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
    # If an image id was given, convert to filename
    if p in h2p:
        p = h2p[p]
    size_x, size_y = p2size[p]

    # Determine the region of the original image we want to capture based on the bounding box.
    row = p2bb.loc[p]
    x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
    dx = x1 - x0
    dy = y1 - y0
    x0 -= dx * crop_margin
    x1 += dx * crop_margin + 1
    y0 -= dy * crop_margin
    y1 += dy * crop_margin + 1
    if x0 < 0:
        x0 = 0
    if x1 > size_x:
        x1 = size_x
    if y0 < 0:
        y0 = 0
    if y1 > size_y:
        y1 = size_y
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy * anisotropy:
        dy = 0.5 * (dx / anisotropy - dy)
        y0 -= dy
        y1 += dy
    else:
        dx = 0.5 * (dy * anisotropy - dx)
        x0 -= dx
        x1 += dx
    # Generate the transformation matrix

    # Read the image, transform to black and white and comvert to numpy array
    img = read_raw_image(p)
    img = img_to_array(img)
    if img.shape[2] == 1:
        img = np.stack((img[..., 0],) * 3, axis=-1)
    y0, y1, x0, x1 = max(0, int(y0)), min(int(y1), img.shape[0]), max(0, int(x0)), min(int(x1), img.shape[1])
    img = img[int(y0):int(y1), int(x0):int(x1), :]
    #    print(x0,x1,y0,y1,img.shape, p)
    img = cv2.resize(img, img_shape[:2])
    img = datagen.standardize([img])[0]
    if augment:
        transform_params = {
            'theta': np.random.uniform(-5, 5),
            'tx': np.random.uniform(-0.05, 0.05),
            'ty': np.random.uniform(-0.05, 0.05),
            'shear': np.random.uniform(-5, 5),
            'zx': np.random.uniform(0.8, 1.05),
            'zy': np.random.uniform(0.8, 1.05)
        }
        img = datagen.apply_transform(img, transform_params)
    # Normalize to zero mean and unit variance
    # img -= np.mean(img, keepdims=True, axis=(0, 1))
    # img /= np.std(img, keepdims=True, axis=(0, 1)) + K.epsilon()
    return img


p = list(tagged.keys())[312]

# model, branch_model, head_model = build_model_standard(64e-5, 0)
print("Using pretrained {}".format(args.model_arch))
model, branch_model, head_model = build_model_pretrained(args.model_arch, 5e-4)
# Find all the whales associated with an image id. It can be ambiguous as duplicate images may have different whale ids.
h2ws = {}
new_whale = 'new_whale'
for p, w in tagged.items():
    if w != new_whale:  # Use only identified whales
        h = p2h[p]
        if h not in h2ws: h2ws[h] = []
        if w not in h2ws[h]: h2ws[h].append(w)
for h, ws in h2ws.items():
    if len(ws) > 1:
        h2ws[h] = sorted(ws)

# For each whale, find the unambiguous images ids.
w2hs = {}
for h, ws in h2ws.items():
    # TODO may need to change this to use ambiguous picture
    if len(ws) == 1:  # Use only unambiguous pictures
        w = ws[0]
        if w not in w2hs: w2hs[w] = []
        if h not in w2hs[w]: w2hs[w].append(h)
for w, hs in w2hs.items():
    if len(hs) > 1:
        w2hs[w] = sorted(hs)

# Find the list of training images, keep only whales with at least two images.
train = []  # A list of training image ids
for hs in w2hs.values():
    if len(hs) > 1:
        train += hs
random.shuffle(train)
train_set = set(train)

w2ts = {}  # Associate the image ids from train to each whale id.
for w, hs in w2hs.items():
    for h in hs:
        if h in train_set:
            if w not in w2ts:
                w2ts[w] = []
            if h not in w2ts[w]:
                w2ts[w].append(h)
for w, ts in w2ts.items():
    w2ts[w] = np.array(ts)

t2i = {}  # The position in train of each training image id
for i, t in enumerate(train):
    t2i[t] = i


class TrainingData(Sequence):
    def __init__(self, score, steps=1000, batch_size=32, img_shape=img_shape, num_threads=4, use_lap=True):
        """
        @param score the cost matrix for the picture matching
        @param steps the number of epoch we are planning with this score matrix
        """
        super(TrainingData, self).__init__()
        self.score = -score if use_lap else None
        self.steps = steps
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.num_threads = num_threads
        self.use_lap = use_lap
        if self.use_lap:
            for ts in w2ts.values():
                idxs = [t2i[t] for t in ts]
                for i in idxs:
                    for j in idxs:
                        self.score[
                            i, j] = 10000.0  # Set a large value for matching whales -- eliminates this potential pairing
        self.on_epoch_end()

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size,) + self.img_shape, dtype=K.floatx())
        b = np.zeros((size,) + self.img_shape, dtype=K.floatx())
        c = np.zeros((size, 1), dtype=K.floatx())
        j = start // 2
        for i in range(0, size, 2):
            a[i, :, :, :] = read_cropped_image(self.match[j][0], True)
            b[i, :, :, :] = read_cropped_image(self.match[j][1], True)
            c[i, 0] = 1  # This is a match
            a[i + 1, :, :, :] = read_cropped_image(self.unmatch[j][0], True)
            b[i + 1, :, :, :] = read_cropped_image(self.unmatch[j][1], True)
            c[i + 1, 0] = 0  # Different whales
            j += 1
        return [a, b], c

    def get_unmatched(self):
        # stupid hack for faster training time
        score_shape = (13623, 13623)
        x = []
        for i in range(score_shape[0]):
            rand_idx = np.random.randint(score_shape[0])
            while rand_idx == i:
                rand_idx = np.random.randint(score_shape[0])
            x.append(rand_idx)
        return np.asarray(x)

    def on_epoch_end(self):
        if self.steps <= 0: return  # Skip this on the last epoch.
        self.steps -= 1
        self.match = []
        self.unmatch = []
        #         print(self.use_lap)
        x = lapjv(self.score)[0] if self.use_lap else self.get_unmatched()
        y = np.arange(len(x), dtype=np.int32)

        # Compute a derangement for matching whales
        for ts in w2ts.values():
            d = ts.copy()
            while True:
                random.shuffle(d)
                if not np.any(ts == d): break
            for ab in zip(ts, d): self.match.append(ab)

        # Construct unmatched whale pairs from the LAP solution.
        for i, j in zip(x, y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i, j)
            assert i != j
            self.unmatch.append((train[i], train[j]))

        # Force a different choice for an eventual next epoch.
        if self.use_lap:
            self.score[x, y] = 10000.0
            self.score[y, x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        # print(len(self.match), len(train), len(self.unmatch), len(train))
        assert len(self.match) == len(train) and len(self.unmatch) == len(train)

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size


score = np.random.random_sample(size=(len(train), len(train)))
data = TrainingData(score, img_shape=img_shape, num_threads=args.lapjv_threads)
(a, b), c = data[0]


# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):
    def __init__(self, data, batch_size=64, verbose=1, TTA=False):
        super(FeatureGen, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.verbose = verbose
        self.TTA = TTA
        if self.verbose > 0: self.progress = tqdm(total=len(self), desc='Features')

    def __getitem__(self, index):
        start = self.batch_size * index
        size = min(len(self.data) - start, self.batch_size)
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        for i in range(size): a[i, :, :, :] = read_cropped_image(self.data[start + i], self.TTA)
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return a

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


# A Keras generator to evaluate on the HEAD MODEL on features already pre-computed.
# It computes only the upper triangular matrix of the cost matrix if y is None.
# x = fknown, y = fsubmit
class ScoreGen(Sequence):
    def __init__(self, x, y=None, batch_size=2048, verbose=1):
        super(ScoreGen, self).__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.verbose = verbose
        if y is None:
            self.y = self.x
            self.ix, self.iy = np.triu_indices(x.shape[0], 1)
        else:
            self.iy, self.ix = np.indices((y.shape[0], x.shape[0]))
            self.ix = self.ix.reshape((self.ix.size,))
            self.iy = self.iy.reshape((self.iy.size,))
        self.subbatch = (len(self.x) + self.batch_size - 1) // self.batch_size
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='Scores')

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.ix))
        a = self.y[self.iy[start:end], :]
        b = self.x[self.ix[start:end], :]
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return [a, b]

    def __len__(self):
        return (len(self.ix) + self.batch_size - 1) // self.batch_size


def set_lr(model, lr):
    K.set_value(model.optimizer.lr, float(lr))


def get_lr(model):
    return K.get_value(model.optimizer.lr)


def score_reshape(score, x, y=None):
    """
    Tranformed the packed matrix 'score' into a square matrix.
    @param score the packed matrix
    @param x the first image feature tensor
    @param y the second image feature tensor if different from x
    @result the square matrix
    """
    if y is None:
        # When y is None, score is a packed upper triangular matrix.
        # Unpack, and transpose to form the symmetrical lower triangular matrix.
        m = np.zeros((x.shape[0], x.shape[0]), dtype=K.floatx())
        m[np.triu_indices(x.shape[0], 1)] = score.squeeze()
        m += m.transpose()
    else:
        m = np.zeros((y.shape[0], x.shape[0]), dtype=K.floatx())
        iy, ix = np.indices((y.shape[0], x.shape[0]))
        ix = ix.reshape((ix.size,))
        iy = iy.reshape((iy.size,))
        m[iy, ix] = score.squeeze()
    return m


def compute_score(verbose=1):
    """
    Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    """
    features = branch_model.predict_generator(FeatureGen(train, verbose=verbose), max_queue_size=12, workers=6,
                                              verbose=0)
    score = head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=6, verbose=0)
    score = score_reshape(score, features)
    return features, score


def make_steps(step, ampl, use_lap=True):
    global w2ts, t2i, steps, features, score, histories

    # shuffle the training pictures
    random.shuffle(train)

    # Map whale id to the list of associated training picture hash value
    w2ts = {}
    for w, hs in w2hs.items():
        for h in hs:
            if h in train_set:
                if w not in w2ts: w2ts[w] = []
                if h not in w2ts[w]: w2ts[w].append(h)
    for w, ts in w2ts.items(): w2ts[w] = np.array(ts)

    # Map training picture hash value to index in 'train' array
    t2i = {}
    for i, t in enumerate(train): t2i[t] = i

    # Compute the match score for each picture pair
    if use_lap:
        features, score = compute_score()

    # Train the model for 'step' epochs
    if use_lap:
        history = model.fit_generator(
            TrainingData(score + ampl * np.random.random_sample(size=score.shape) if use_lap else None, steps=step,
                         batch_size=args.batch_size,
                         use_lap=use_lap,
                         img_shape=img_shape, num_threads=args.lapjv_threads),
            initial_epoch=steps, epochs=steps + step, max_queue_size=12, workers=6, verbose=1).history
    else:
        for st in range(step):
            history = model.fit_generator(
                TrainingData(score + ampl * np.random.random_sample(size=score.shape) if use_lap else None, steps=1,
                             batch_size=args.batch_size,
                             use_lap=use_lap,
                             img_shape=img_shape, num_threads=args.lapjv_threads),
                initial_epoch=steps + st, epochs=1, max_queue_size=12, workers=6, verbose=1).history
    steps += step

    # Collect history data
    history['epochs'] = steps
    history['ms'] = np.mean(score) if use_lap else 0
    history['lr'] = get_lr(model)
    print(history['epochs'], history['lr'], history['ms'])
    histories.append(history)


histories = []
steps = 0

if isfile(args.model_path):
    print("Loading model from {}".format(args.model_path))
    try:
        tmp = keras.models.load_model(args.model_path)
        model.set_weights(tmp.get_weights())
    except:
        model.load_weights(args.model_path)
if not args.predict_only:
    print("Training...")
    optim = Adam(lr=5e-4)
    # epoch -> 10
    for layer in branch_model.layers:
        layer.trainable = False
    model.compile(optim, loss='binary_crossentropy', metrics=['acc'])
    make_steps(3, 1000, use_lap=True)

    for layer in branch_model.layers:
        layer.trainable = True

    optim = Adam(lr=2e-4)
    model.compile(optim, loss='binary_crossentropy', metrics=['acc'])
    make_steps(7, 1000, use_lap=True)

    ampl = 100.0
    model.save(args.model_path)

    for _ in range(10):
        print('noise ampl.  = ', ampl)
        make_steps(5, ampl, use_lap=True)
        model.save(args.model_path)
        ampl = max(1.0, 100 ** -0.1 * ampl)

    set_lr(model, 1e-4)

    for _ in range(8):
        make_steps(5, 1.0, use_lap=True)
        model.save(args.model_path)

    for _ in range(10):
        make_steps(5, 1.0, use_lap=False)
        model.save(args.model_path)
    
    set_lr(model, 16e-5)
    for _ in range(10):
        make_steps(5, 0.5, use_lap=False)
        model.save(args.model_path)

    # set_lr(model, 4e-5)
    # for _ in range(8):
    #     make_steps(5, 0.25)
    #     model.save(args.model_path)
    #
    # set_lr(model, 1e-5)
    # for _ in range(2): make_steps(5, 0.25)
    # model.save(args.model_path)
    #
    # weights = model.get_weights()
    # model, branch_model, head_model = build_model_pretrained(args.model_arch, 5e-4)
    # model.set_weights(weights)
    #
    # for _ in range(10): make_steps(5, 1.0)
    # model.save(args.model_path)
    #
    # set_lr(model, 16e-5)
    # for _ in range(10): make_steps(5, 0.5)
    # model.save(args.model_path)
    #
    # set_lr(model, 4e-5)
    # for _ in range(8): make_steps(5, 0.25)
    # model.save(args.model_path)
    #
    # set_lr(model, 1e-5)
    # for _ in range(2): make_steps(5, 0.25)
    # model.save(args.model_path)


def prepare_submission(threshold, filename):
    """
    Generate a Kaggle submission file.
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    """
    vtop = 0
    vhigh = 0
    pos = [0, 0, 0, 0, 0, 0]
    with open(filename, 'wt', newline='\n') as f:
        f.write('Image,Id\n')
        for i, p in enumerate(tqdm(submit)):
            t = []
            s = set()
            a = score[i, :]
            for j in list(reversed(np.argsort(a))):
                h = known[j]
                if a[j] < threshold and new_whale not in s:
                    pos[len(t)] += 1
                    s.add(new_whale)
                    t.append(new_whale)
                    if len(t) == 5: break;
                for w in h2ws[h]:
                    assert w != new_whale
                    if w not in s:
                        if a[j] > 1.0:
                            vtop += 1
                        elif a[j] >= threshold:
                            vhigh += 1
                        s.add(w)
                        t.append(w)
                        if len(t) == 5: break;
                if len(t) == 5: break;
            if new_whale not in s: pos[5] += 1
            assert len(t) == 5 and len(s) == 5
            f.write(p + ',' + ' '.join(t[:5]) + '\n')
    return vtop, vhigh, pos


# Find elements from training sets not 'new_whale'
h2ws = {}
for p, w in tagged.items():
    if w != new_whale:  # Use only identified whales
        h = p2h[p]
        if h not in h2ws:
            h2ws[h] = []
        if w not in h2ws[h]:
            h2ws[h].append(w)
known = sorted(list(h2ws.keys()))

# Dictionary of picture indices
h2i = {}
for i, h in enumerate(known):
    h2i[h] = i

# Evaluate the model.
fknown = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=1)
fsubmit = branch_model.predict_generator(FeatureGen(submit), max_queue_size=20, workers=10, verbose=1)
score = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=1)
score = score_reshape(score, fknown, fsubmit)

if args.tta > 0:
    for i in tqdm(range(args.tta), unit=" Fold"):
        fsubmit = branch_model.predict_generator(FeatureGen(submit, TTA=True), max_queue_size=20, workers=10, verbose=0)
        score += score_reshape(
            head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
            , fknown, fsubmit)
    score /= (args.tta + 1)
# Generate the subsmission file.
prepare_submission(0.99, 'submission.csv')
