import logging
import random
from itertools import compress

import scipy.io as sio
from torch.utils.data import Dataset

from .datautils import *

logger = logging.getLogger(__name__)


class MyDataset(Dataset):

    structure = {
        'training': {
            'images': 'img',
            'gt': 'gt',
        },
        'test': {
            'images': 'img',
        },
    }

    def __init__(self, data_root, type='training'):
        data_root = pathlib.Path(data_root)

        if type == 'test':
            logger.warning('Our dataset does not contain test ground truth. Fall back to training instead.')

        self.structure = MyDataset.structure                            # 数据的路径
        self.imagesRoot = data_root / self.structure[type]['images']    # 图像目录
        self.gtRoot = data_root / self.structure[type]['gt']            # 标注目录
        self.images, self.bboxs, self.transcripts = self.__loadGT()     # 导入所有数据

    def __loadGT(self):
        all_bboxs = []
        all_texts = []
        all_images = []
        for image in self.imagesRoot.glob('*.jpg'):
            all_images.append(image)
            gt = self.gtRoot / image.with_suffix('.txt').name # 直接将.jpg的文件改成.txt就是对应的标注了
            with gt.open(mode='r') as f:
                bboxes = []
                texts = []
                for line in f: # 一行是“x1,y1,x2,y2,x3,y3,x4,y4,语言,内容”
                    text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',')
                    x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, text[:8]))
                    bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    transcript = text[9] # 只要内容，不管语言
                    bboxes.append(bbox)
                    texts.append(transcript)
                bboxes = np.array(bboxes)
                all_bboxs.append(bboxes)
                all_texts.append(texts)
                
        return all_images, all_bboxs, all_texts # 图像路径，bbox坐标，文本

    def __getitem__(self, index): # 返回经过transform处理后的图像、标注
        imageName = self.images[index]
        bboxes = self.bboxs[index]  # num_words * 8
        transcripts = self.transcripts[index]

        try:
            return self.__transform((imageName, bboxes, transcripts))
        except Exception as e: # 无法预处理则换一个样本
            # tb.print_exc(file=sys.stdout)
            return self.__getitem__(torch.tensor(np.random.randint(0, len(self))))

    def __len__(self):
        return len(self.images)

    def __transform(self, gt, input_size=512, random_scale=np.array([0.5, 1, 2.0, 3.0]),
                    background_ratio=3. / 8):
        """
        preprocessing of the input data，random resize + random crop
        
        :param gt: image path (str), wordBBoxes (2 * 4 * num_words), transcripts (multiline)

        :return: imagePath, images, score_maps, geo_maps, training_masks, transcripts, rectangles
        """

        imagePath, wordBBoxes, transcripts = gt
        im = cv2.imread(imagePath.as_posix())
        # wordBBoxes = np.expand_dims(wordBBoxes, axis = 2) if (wordBBoxes.ndim == 2) else wordBBoxes
        # _, _, numOfWords = wordBBoxes.shape
        numOfWords = len(wordBBoxes)
        text_polys = wordBBoxes  # num_words * 4 * 2，每个词语有四个角的坐标共8个参数
        transcripts = np.array([word for line in transcripts for word in line.split()])
        text_tags = [True if (tag == '*' or tag == '###') else False for tag in transcripts]  # ignore '###'

        if numOfWords == len(transcripts): # 只有bbox数量与文本数量相同才可以进行预处理
            h, w, _ = im.shape
            text_polys, transcripts, text_tags = check_and_validate_polys(text_polys, transcripts, text_tags, (h, w))

            rd_scale = np.random.choice(random_scale)
            im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale) # 随机缩放图像
            text_polys *= rd_scale

            rectangles = []

            # print rd_scale
            # random crop a area from image
            if np.random.rand() < background_ratio: # 有一定概率不考虑rbox
                # crop background
                im, text_polys, text_tags,transcripts, selected_poly = crop_area(im, text_polys,transcripts, text_tags, crop_background=True)
                if text_polys.shape[0] > 0:
                    raise RuntimeError('cannot find background')
                    
                # pad and resize image，先将图像补成方形，再变换成原尺寸
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = cv2.resize(im_padded, dsize=(input_size, input_size))
                
                score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                geo_map_channels = 5
                geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32) # 5通道feature，参考论文
                training_mask = np.ones((input_size, input_size), dtype=np.uint8) # 最后一通道为mask
            else:
                im, text_polys, text_tags,transcripts, selected_poly = crop_area(im, text_polys, text_tags,transcripts, crop_background=False)
                if text_polys.shape[0] == 0:
                    raise RuntimeError('cannot find background')
                h, w, _ = im.shape

                # 将图像补成方形
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = im_padded
                # 变换成原尺寸，同第一种情况
                new_h, new_w, _ = im.shape
                resize_h = input_size
                resize_w = input_size
                im = cv2.resize(im, dsize=(resize_w, resize_h))
                
                resize_ratio_3_x = resize_w / float(new_w) # w方向上与原图的比值
                resize_ratio_3_y = resize_h / float(new_h) # h方向上与原图的比值
                text_polys[:, :, 0] *= resize_ratio_3_x # 将bbox的w方向上resize
                text_polys[:, :, 1] *= resize_ratio_3_y # 将bbox的h方向上resize
                new_h, new_w, _ = im.shape
                score_map, geo_map, training_mask, rectangles = generate_rbox((new_h, new_w), text_polys, text_tags)

            step_size = int(input_size // 128)
            # 将各种score_maps, geo_maps, training_masks变为128*128
            images = im[:, :, ::-1].astype(np.float32)  # bgr -> rgb
            score_maps = score_map[::step_size, ::step_size, np.newaxis].astype(np.float32)
            geo_maps = geo_map[::step_size, ::step_size, :].astype(np.float32)
            training_masks = training_mask[::step_size, ::step_size, np.newaxis].astype(np.float32)

            transcripts = [transcripts[i] for i in selected_poly] # 只要crop出来的区域的那些文本
            mask = [not (word == '*' or word == '###') for word in transcripts] # 舍弃掉那些不关注的词语对应的mask
            transcripts = list(compress(transcripts, mask))
            rectangles = list(compress(rectangles, mask))  # [ [pt1, pt2, pt3, pt3],  ]

            assert len(transcripts) == len(rectangles)  # make sure length of transcipts equal to length of boxes
            if len(transcripts) == 0:
                raise RuntimeError('No text found.')

            return imagePath, images, score_maps, geo_maps, training_masks, transcripts, rectangles
        else:
            raise TypeError('Number of bboxes is inconsistent with number of transcripts ')


class ICDAR(Dataset):
    structure = {
        '2015': {
            'training': {
                'images': 'ch4_training_images',
                'gt': 'ch4_training_localization_transcription_gt',
                'voc_per_image': 'ch4_training_vocabularies_per_image',
                'voc_all': 'ch4_training_vocabulary.txt'
            },
            'test': {
                'images': 'ch4_test_images',
                'gt': 'Challenge4_Test_Task4_GT',
                'voc_per_image': 'ch4_test_vocabularies_per_image',
                'voc_all': 'ch4_test_vocabulary.txt'
            },
            'voc_generic': 'GenericVocabulary.txt'
        },
        '2013': {
            'training': {
                'images': 'ch2_training_images',
                'gt': 'ch2_training_localization_transcription_gt',
                'voc_per_image': 'ch2_training_vocabularies_per_image',
                'voc_all': 'ch2_training_vocabulary.txt'
            },
            'test': {
                'images': 'Challenge2_Test_Task12_Images',
                'voc_per_image': 'ch2_test_vocabularies_per_image',
                'voc_all': 'ch4_test_vocabulary.txt'
            },
            'voc_generic': 'GenericVocabulary.txt'
        },

    }

    def __init__(self, data_root, year='2013', type='training'):
        data_root = pathlib.Path(data_root)

        if year == '2013' and type == 'test':
            logger.warning('ICDAR 2013 does not contain test ground truth. Fall back to training instead.')

        self.structure = ICDAR.structure[year]                          # 这一年的数据的路径
        self.imagesRoot = data_root / self.structure[type]['images']    # 图像文件夹
        self.gtRoot = data_root / self.structure[type]['gt']            # 标注文件夹
        self.images, self.bboxs, self.transcripts = self.__loadGT()

    def __loadGT(self):
        all_bboxs = []
        all_texts = []
        all_images = []
        for image in self.imagesRoot.glob('*.jpg'):
            all_images.append(image)
            gt = self.gtRoot / image.with_name('gt_{}'.format(image.stem)).with_suffix('.txt').name
            with gt.open(mode='r') as f:
                bboxes = []
                texts = []
                for line in f:
                    text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',')
                    x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, text[:8]))
                    bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    transcript = text[8]
                    bboxes.append(bbox)
                    texts.append(transcript)
                bboxes = np.array(bboxes)
                all_bboxs.append(bboxes)
                all_texts.append(texts)
        return all_images, all_bboxs, all_texts

    def __getitem__(self, index): # 返回经过transform处理后的图像
        imageName = self.images[index]
        bboxes = self.bboxs[index]  # num_words * 8
        transcripts = self.transcripts[index]

        try:
            return self.__transform((imageName, bboxes, transcripts))
        except Exception as e:
            # tb.print_exc(file=sys.stdout)
            return self.__getitem__(torch.tensor(np.random.randint(0, len(self))))

    def __len__(self):
        return len(self.images)

    def __transform(self, gt, input_size=512, random_scale=np.array([0.5, 1, 2.0, 3.0]),
                    background_ratio=3. / 8):
        """

        :param gt: image path (str), wordBBoxes (2 * 4 * num_words), transcripts (multiline)

        :return:
        """

        imagePath, wordBBoxes, transcripts = gt
        im = cv2.imread(imagePath.as_posix())
        # wordBBoxes = np.expand_dims(wordBBoxes, axis = 2) if (wordBBoxes.ndim == 2) else wordBBoxes
        # _, _, numOfWords = wordBBoxes.shape
        numOfWords = len(wordBBoxes)
        text_polys = wordBBoxes  # num_words * 4 * 2
        transcripts = np.array([word for line in transcripts for word in line.split()])
        text_tags = [True if (tag == '*' or tag == '###') else False for tag in transcripts]  # ignore '###'

        if numOfWords == len(transcripts):
            h, w, _ = im.shape
            text_polys, transcripts, text_tags = check_and_validate_polys(text_polys, transcripts, text_tags, (h, w))

            rd_scale = np.random.choice(random_scale)
            im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
            text_polys *= rd_scale

            rectangles = []

            # print rd_scale
            # random crop a area from image
            if np.random.rand() < background_ratio:
                # crop background
                im, text_polys, text_tags,transcripts, selected_poly = crop_area(im, text_polys,transcripts, text_tags, crop_background=True)
                if text_polys.shape[0] > 0:
                    # cannot find background
                    raise RuntimeError('cannot find background')
                # pad and resize image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = cv2.resize(im_padded, dsize=(input_size, input_size))
                score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                geo_map_channels = 5
                geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                training_mask = np.ones((input_size, input_size), dtype=np.uint8)
            else:
                im, text_polys, text_tags,transcripts, selected_poly = crop_area(im, text_polys, text_tags,transcripts, crop_background=False)
                if text_polys.shape[0] == 0:
                    raise RuntimeError('cannot find background')
                h, w, _ = im.shape

                # pad the image to the training input size or the longer side of image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = im_padded
                # resize the image to input size
                new_h, new_w, _ = im.shape
                resize_h = input_size
                resize_w = input_size
                im = cv2.resize(im, dsize=(resize_w, resize_h))
                resize_ratio_3_x = resize_w / float(new_w)
                resize_ratio_3_y = resize_h / float(new_h)
                text_polys[:, :, 0] *= resize_ratio_3_x
                text_polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im.shape
                score_map, geo_map, training_mask, rectangles = generate_rbox((new_h, new_w), text_polys, text_tags)

            step_size = int(input_size // 128)

            images = im[:, :, ::-1].astype(np.float32)  # bgr -> rgb
            score_maps = score_map[::step_size, ::step_size, np.newaxis].astype(np.float32)
            geo_maps = geo_map[::step_size, ::step_size, :].astype(np.float32)
            training_masks = training_mask[::step_size, ::step_size, np.newaxis].astype(np.float32)

            transcripts = [transcripts[i] for i in selected_poly]
            mask = [not (word == '*' or word == '###') for word in transcripts]
            transcripts = list(compress(transcripts, mask))
            rectangles = list(compress(rectangles, mask))  # [ [pt1, pt2, pt3, pt3],  ]

            assert len(transcripts) == len(rectangles)  # make sure length of transcipts equal to length of boxes
            if len(transcripts) == 0:
                raise RuntimeError('No text found.')

            return imagePath, images, score_maps, geo_maps, training_masks, transcripts, rectangles
        else:
            raise TypeError('Number of bboxes is inconsistent with number of transcripts ')


class SynthTextDataset(Dataset):

    def __init__(self, data_root):
        self.dataRoot = pathlib.Path(data_root)
        if not self.dataRoot.exists():
            raise FileNotFoundError('Dataset folder is not exist.')

        self.targetFilePath = self.dataRoot / 'gt.mat'
        if not self.targetFilePath.exists():
            raise FileExistsError('Target file is not exist.')
        targets = {}
        sio.loadmat(self.targetFilePath, targets, squeeze_me=True, struct_as_record=False,
                    variable_names=['imnames', 'wordBB', 'txt'])

        self.imageNames = targets['imnames']
        self.wordBBoxes = targets['wordBB']
        self.transcripts = targets['txt']

    def __getitem__(self, index):
        """

        :param index:
        :return:
            imageName: path of image
            wordBBox: bounding boxes of words in the image
            transcript: corresponding transcripts of bounded words
        """
        imageName = self.imageNames[index]
        wordBBoxes = self.wordBBoxes[index]  # 2 * 4 * num_words
        transcripts = self.transcripts[index]

        try:
            return self.__transform((imageName, wordBBoxes, transcripts))
        except:
            return self.__getitem__(np.random.randint(0, len(self)))

    def __len__(self):
        return len(self.imageNames)

    def __transform(self, gt, input_size=512, random_scale=np.array([0.5, 1, 2.0, 3.0]),
                    background_ratio=3. / 8):
        '''

        :param gt: iamge path (str), wordBBoxes (2 * 4 * num_words), transcripts (multiline)

        :return:
        '''

        imagePath, wordBBoxes, transcripts = gt
        im = cv2.imread((self.dataRoot / imagePath).as_posix())
        wordBBoxes = np.expand_dims(wordBBoxes, axis=2) if (wordBBoxes.ndim == 2) else wordBBoxes
        _, _, numOfWords = wordBBoxes.shape
        text_polys = wordBBoxes.reshape([8, numOfWords], order='F').T  # num_words * 8
        text_polys = text_polys.reshape(numOfWords, 4, 2)  # num_of_words * 4 * 2
        transcripts = [word for line in transcripts for word in line.split()]
        text_tags = np.zeros(numOfWords)  # 1 to ignore, 0 to hold

        if numOfWords == len(transcripts):
            h, w, _ = im.shape
            text_polys, transcripts, text_tags = check_and_validate_polys(text_polys, transcripts, text_tags, (h, w))

            rd_scale = np.random.choice(random_scale)
            im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
            text_polys *= rd_scale

            # print rd_scale
            # random crop a area from image
            if np.random.rand() < background_ratio:
                # crop background
                im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
                # if text_polys.shape[0] > 0:
                #     # cannot find background
                #     pass
                # pad and resize image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = cv2.resize(im_padded, dsize=(input_size, input_size))
                score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                geo_map_channels = 5
                #                     geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
                geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                training_mask = np.ones((input_size, input_size), dtype=np.uint8)
            else:
                im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                # if text_polys.shape[0] == 0:
                #     pass
                h, w, _ = im.shape

                # pad the image to the training input size or the longer side of image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = im_padded
                # resize the image to input size
                new_h, new_w, _ = im.shape
                resize_h = input_size
                resize_w = input_size
                im = cv2.resize(im, dsize=(resize_w, resize_h))
                resize_ratio_3_x = resize_w / float(new_w)
                resize_ratio_3_y = resize_h / float(new_h)
                text_polys[:, :, 0] *= resize_ratio_3_x
                text_polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im.shape
                score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)

            # predict 出来的feature map 是 128 * 128， 所以 gt 需要取 /4 步长
            images = im[:, :, ::-1].astype(np.float32)  # bgr -> rgb
            score_maps = score_map[::4, ::4, np.newaxis].astype(np.float32)
            geo_maps = geo_map[::4, ::4, :].astype(np.float32)
            training_masks = training_mask[::4, ::4, np.newaxis].astype(np.float32)

            return images, score_maps, geo_maps, training_masks, transcripts

            # return images, score_maps, geo_maps, training_masks
        else:
            raise TypeError('Number of bboxes is inconsist with number of transcripts ')
