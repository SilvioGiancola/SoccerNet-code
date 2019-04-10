from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os

import h5py
import numpy as np
import tensorflow as tf

from skimage.io import imread

import skvideo.io
import json
from tqdm import tqdm
import time

import i3d
#from PIL import Image

_NUM_FRAMES = 64
_IMAGE_SIZE = 224
_NUM_CLASSES = 400

_CHECKPOINT_PATHS = {
    'rgb': 'kinetics-i3d/data/checkpoints/rgb_scratch/model.ckpt',
    'rgb_imagenet': 'kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt'
}

# Make sure you set the verbosity level at your best convinience.
tf.logging.set_verbosity(tf.logging.INFO)


def scale_image(x, min_x=0, max_x=255):
    x_norm = 2*((x-min_x)/(max_x-min_x)) - 1
    return x_norm.astype(np.float32)


def center_crop(img, cropx=_IMAGE_SIZE, cropy=_IMAGE_SIZE):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


def load_and_preprocess_input(videoPath, 
    num_frames_per_clip=_NUM_FRAMES, image_size=_IMAGE_SIZE,
    filename_extension='png'):

    print(videoPath)

    # get FPS
    metadata = skvideo.io.ffprobe(videoPath)
    fps_meta = json.dumps(metadata["video"]['@avg_frame_rate'], indent=4).replace('"','')
    fps = int(fps_meta.split('/')[0])/int(fps_meta.split('/')[1])

    # get DURATION
    for key in metadata["video"]["tag"]:
        if (key["@key"] == "DURATION"):
            time_str = key["@value"]
            h, m, s = time_str.split(".")[0].split(":")
            break;
    duration = int(h)*60*60 + int(m)*60 + int(s)

    # print info
    print(videoPath)
    print("duration=", duration)
    print("fps=", fps)
    nb_frames = int(duration*fps)
    print("nb_frames=", nb_frames)

    index_frame = 0
    vid = []
    videogen = skvideo.io.vreader(videoPath, num_frames=int(nb_frames), backend='ffmpeg')
    for frame in videogen:
        vid.append(center_crop(scale_image(frame)))
        index_frame = index_frame + 1      
        if (index_frame >= nb_frames):         
            break

    # stack num_frames_per_clip frames
    for index_frame in range(len(vid)-num_frames_per_clip):
    #for index_frame in range(200):
        if (int((index_frame))%(fps*0.5) < 1):
            # print(num_frames_per_clip, index_frame)
            # im = Image.fromarray(vid[index_frame])
            # im.save("images/frame" + str(index_frame) + ".jpeg")
            rgb_input = np.stack(vid[index_frame:index_frame+num_frames_per_clip])
            yield rgb_input[np.newaxis, ...]
          


def main(video_dir, feature_dir, imagenet_pretrained=True,overwrite=False):

    print(video_dir)
    # Graph definition:
    rgb_input = tf.placeholder(tf.float32,
        shape=(1, _NUM_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(
            _NUM_CLASSES, spatial_squeeze=True, final_endpoint='PreLogits')
        rgb_prelogits, _ = rgb_model(
            rgb_input, is_training=False, dropout_keep_prob=1.0)
    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    # Session
    with tf.Session() as sess:
        # Restoring checkpoint.
        if imagenet_pretrained:
            rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
        else:
            rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
        tf.logging.info('RGB checkpoint restored')


        if ( not os.path.exists(os.path.join(feature_dir,"1_I3D.npy")) or overwrite):

            print('going for Half 1')
            # Densely extracting PreLogit features.
            feat_buffer = []
            cnt = 1
            video_path = os.path.join(video_dir,"1.mkv")
            print(video_path)
            for this_input in tqdm(load_and_preprocess_input(video_path)):
                feat_i = sess.run(
                    rgb_prelogits, feed_dict={rgb_input: this_input})
                feat_buffer.append(feat_i.squeeze().mean(axis=0))
                # print('processed %s' % cnt)
                cnt += 1


            print("exporting in vstack")
            feat_buffer_vstack = np.array(feat_buffer)
          
            print("saving in NPY")
            np.save(os.path.join(feature_dir,"1_I3D.npy"), feat_buffer_vstack)
            print('Done')
        else:
            print('Half 1 already exists')



        if ( not os.path.exists(os.path.join(feature_dir,"2_I3D.npy")) or overwrite):

            print('going for Half 2')
            # Densely extracting PreLogit features.
            feat_buffer = []
            cnt = 1
            video_path = os.path.join(video_dir,"2.mkv")
            for this_input in load_and_preprocess_input(video_path):
                feat_i = sess.run(
                    rgb_prelogits, feed_dict={rgb_input: this_input})
                feat_buffer.append(feat_i.squeeze().mean(axis=0))
                print('processed %s' % cnt)
                cnt += 1


            print("exporting in array")
            feat_buffer_vstack = np.array(feat_buffer)
           

            print("saving in NPY")
            np.save(os.path.join(feature_dir,"2_I3D.npy"), feat_buffer_vstack)
            print('Done')

        else:
            print('Half 2 already exists')

if __name__ == '__main__':
    description = 'I3D Spatial Stream Feature Extractor'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('video_list_dir', type=str, 
        help='Folder containing a batch of videos.')
    p.add_argument('feature_dir', type=str,
        help='Folder where features will be saved.')
    p.add_argument('--jobid', type=int, default=-1,
        help='Job identifier retrieved from SLURM.')
    p.add_argument('--GPU',        help='ID of the GPU to use' ,   required=False, type=int,   default=-1)
    p.add_argument('--overwrite', help='overwrite features',    required=False, action="store_true")

    args = p.parse_args()


    import os 
    if (args.GPU >= 0):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    if not os.path.exists(args.feature_dir):
        raise IOError('Please create `feature_dir` folder')

    directories = []
    # if (os.path.exists(os.path.join(args.video_list_dir,"listgame.npy"))):
    #     print("Loading List Games")
    #     directories = np.load(os.path.join(args.video_list_dir,"listgame.npy"))
    # else:
    for Championship in next(os.walk(args.video_list_dir))[1]:
        for Year in next(os.walk(os.path.join(args.video_list_dir, Championship)))[1]:
            for Game in next(os.walk(os.path.join(args.video_list_dir, Championship, Year)))[1]:
                Game_FullPath = os.path.join(args.video_list_dir, Championship, Year, Game)
                
                directories.append( Game_FullPath )

    directories.sort()
        # np.save(os.path.join(args.video_list_dir,"listgame.npy"), np.array(directories))

    # video_lst = sorted(glob.glob(os.path.join(args.video_list_dir, '*')))

    start_time = time.time()    

    if (args.jobid == -1):
        for directory in directories:
            args.video_list_dir = os.path.join(args.video_list_dir, directory)
            args.feature_dir = os.path.join(args.feature_dir, directory)
            # feature_filename = os.path.join(args.feature_dir, directory, 'i3D.hdf5')
            # if not os.path.exists(feature_filename):
            main(args.video_list_dir, args.feature_dir, overwrite=args.overwrite)
    else:
        directory = directories[args.jobid]
        args.video_list_dir = os.path.join(args.video_list_dir, directory)
        args.feature_dir = os.path.join(args.feature_dir, directory)
        # feature_filename = os.path.join(args.feature_dir, directory, 'i3D.hdf5')
        # if not os.path.exists(feature_filename):
        main(args.video_list_dir, args.feature_dir, overwrite=args.overwrite)
    

    print("elapsed time =", time.time() - start_time, "s")
