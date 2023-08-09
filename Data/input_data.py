import random
import time
import os
import torch.multiprocessing


def culen(file):
    for dirpath, dirnames, filenames in os.walk(file):
        return len(filenames)


def get_pic(file_path, clips_num, interval=1, double=False,test=False, batch_index=0, st=0,cam1='cam1',cam2='cam2'):
    """

    :param file_path:文件存放路径
    :param clips_num:一个动作多少帧
    :param interval:每间隔多少帧取一帧（默认为1）
    :param double:是否读入双角度图片(默认为false)
    :param test:是否测试
    :return:
    """
    n = st
    with open(file_path, 'r') as lines:
        cam1_input = []
        cam2_input = []
        labels = []
        lines = list(lines)
        long = len(lines)

        while batch_index < long:

            line = lines[batch_index].strip('\n').split()
            dirname = os.path.join(line[0])
            tmp_label = line[1]
            cam1_path = os.path.join(dirname, cam1)
            lo = culen(cam1_path)  # 计算一个动作的总长度
            ma = lo - clips_num * interval  # 计算可以分解成多少个小的clips

            while n < ma:
                rgb_input_path = get_picture(cam1_path, clips_num, interval, s_index = n)
                # print("Loading a video clip from {}...".format(cam1_path))
                if double:
                    cam2_path = os.path.join(dirname, cam2)
                    # print("Loading a video clip from {}...".format(cam2_path))
                    cam2_input_path = get_picture(cam2_path, clips_num, interval, s_index = n)
                    cam2_input.append(cam2_input_path)
                cam1_input.append(rgb_input_path)
                labels.append(int(tmp_label))
                n = n + 20

            batch_index = batch_index + 1
            n = 0
        # print('---------------------------------load finish-----------------------------------')

    return cam1_input, cam2_input, labels,


def get_picture(filename, clips_num, interval=1, s_index=-1):

    img_path = []
    filenames = ''

    if s_index < 0:
        s_index = random.randint(0, len(filenames) - clips_num * interval)

    for parent, dirnames, filenames in os.walk(filename):
        filenames = sorted(filenames)
        if len(filenames) == 0:
            print('DATA_ERRO: %s' % filename)
            return [], s_index

        if (len(filenames) - s_index) <= clips_num * interval:
            # num = 1
            # while len(filenames) - s_index + num < clips_num * interval:
            #     print("11",s_index)
            #     image_name = str(filename) + '/' + str(filenames[s_index + num])
            #     img_path.append(image_name)
            img_path = [str(filename) + '/'+ filenames[s_index+i] for i in range((len(filenames) - s_index))]
                # num += 1
            leave = clips_num - len(img_path)
            for i in range(leave):
                image_name = str(filename) + '/' + str(filenames[len(filenames)-1])
                img_path.append(image_name)

        else:
            for i in range(clips_num):
                image_name = str(filename) + '/' + str(filenames[s_index + i*interval ])
                img_path.append(image_name)
        # print("-------------------",len(img_path))
        if len(img_path) != clips_num:
            print("error")
    return img_path


