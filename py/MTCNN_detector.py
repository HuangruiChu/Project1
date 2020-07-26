#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import cv2
import torchvision
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from Utils import nms,convert_to_square,IoU, img_normalization
import torch
from torchvision import transforms
transform = transforms.ToTensor()
from torch.autograd.variable import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_image_to_tensor(image):
    """convert an image to pytorch tensor

        Parameters:
        ----------
        image: numpy array , h * w * c

        Returns:
        -------
        image_tensor: pytorch.FloatTensor, c * h * w
        """
    # image = image.astype(np.float32)
    return transform(image)
    # return transform(image)
def convert_chwTensor_to_hwcNumpy(tensor):
    """convert a group images pytorch tensor(count * c * h * w) to numpy array images(count * h * w * c)
            Parameters:
            ----------
            tensor: numpy array , count * c * h * w

            Returns:
            -------
            numpy array images: count * h * w * c
            """

    if isinstance(tensor, Variable):
        return np.transpose(tensor.data.numpy(), (0,2,3,1))
    elif isinstance(tensor, torch.FloatTensor):
        return np.transpose(tensor.numpy(), (0,2,3,1))
    else:
        raise Exception("covert b*c*h*w tensor to b*h*w*c numpy error.This tensor must have 4 dimension.")


# In[ ]:


class MtcnnDetector(object):
    """
        P,R,O net face detection and landmarks align
    """
    def  __init__(self,
                 pnet = None,
                 rnet = None,
                 onet = None,
                 min_face_size=12,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709,
                 ):

        self.pnet_detector = pnet
        self.rnet_detector = rnet
        self.onet_detector = onet
        self.min_face_size = min_face_size
        self.stride=stride
        self.thresh = threshold
        self.scale_factor = scale_factor

    def generate_bounding_box(self, map, reg, scale, threshold):
        """
            generate bbox from feature map
        Parameters:
        ----------
            map: numpy array , n x m x 1
                detect score for each position
            reg: numpy array , n x m x 4
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        stride = 2
        cellsize = 12 # receptive field

        t_index = np.where(map[:,:,0] > threshold)
        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        # choose bounding box whose socre are larger than threshold
        dx1, dy1, dx2, dy2 = [reg[0, t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = map[t_index[0], t_index[1], 0]
        # hence t_index[1] means column, t_index[1] is the value of x
        # hence t_index[0] means row, t_index[0] is the value of y
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),            # x1 of prediction box in original image
                                 np.round((stride * t_index[0]) / scale),            # y1 of prediction box in original image
                                 np.round((stride * t_index[1] + cellsize) / scale), # x2 of prediction box in original image
                                 np.round((stride * t_index[0] + cellsize) / scale), # y2 of prediction box in original image
                                # reconstruct the box in original image
                                 score,
                                 reg,
                                 # landmarks
                                 ])

        return boundingbox.T


    def resize_image(self, img, scale):
        """
            resize image and transform dimention to [batchsize, channel, height, width]
        Parameters:
        ----------
            img: numpy array , height x width x channel
                input image, channels in BGR order here
            scale: float number
                scale factor of resize operation
        Returns:
        -------
            transformed image tensor , 1 x channel x height x width
        """
        height, width, channels = img.shape
        new_height = int(height * scale)     # resized new height
        new_width = int(width * scale)       # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
        return img_resized


    def pad(self, bboxes, w, h):
        """
            pad the the boxes
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
        # width and height
        tmpw = (bboxes[:, 2] - bboxes[:, 0] + 1).astype(np.int32)
        tmph = (bboxes[:, 3] - bboxes[:, 1] + 1).astype(np.int32)
        numbox = bboxes.shape[0]

        dx = np.zeros((numbox, ))
        dy = np.zeros((numbox, ))
        edx, edy  = tmpw.copy()-1, tmph.copy()-1
        # x, y: start point of the bbox in original image
        # ex, ey: end point of the bbox in original image
        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w-1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h-1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list


    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array
            input image array
            one batch

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """
        # original wider face data
        im = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
        h, w, c = im.shape
        net_size = 12
        current_scale = float(net_size) / self.min_face_size    # find initial scale
        #print('imgshape:{0}, current_scale:{1}'.format(im.shape, current_scale))
        im_resized = self.resize_image(im, current_scale) # scale = 1.0
        current_height, current_width, _ = im_resized.shape
        # fcn
        all_boxes = list()
        while min(current_height, current_width) > net_size:
            #print('current:',current_height, current_width)
            feed_imgs = []
            image_tensor =convert_image_to_tensor(im_resized)
            feed_imgs.append(image_tensor)
            feed_imgs = torch.stack(feed_imgs).to(device)
            # self.pnet_detector is a trained pnet torch model
            # receptive field is 12×12
            # 12×12 --> score
            # 12×12 --> bounding box
            cls_map, reg = self.pnet_detector(feed_imgs)

            cls_map_np = convert_chwTensor_to_hwcNumpy(cls_map.cpu())
            reg_np = convert_chwTensor_to_hwcNumpy(reg.cpu())
            
            # boxes = [x1, y1, x2, y2, score, reg]
            boxes = self.generate_bounding_box(cls_map_np[ 0, :, :], reg_np, current_scale, self.thresh[0])
           
            # generate pyramid images
            current_scale *= self.scale_factor # self.scale_factor = 0.709
            im_resized = self.resize_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue

            # non-maximum suppresion
            keep = nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None
        all_boxes = np.vstack(all_boxes)
        
        # merge the detection from first stage
        keep = nms(all_boxes[:, 0:5], 0.6, 'Union')
        all_boxes = all_boxes[keep]
        # boxes = all_boxes[:, :5]

        # x2 - x1
        # y2 - y1
        bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # landmark_keep = all_boxes[:, 9:].reshape((5,2))
        boxes = np.vstack([all_boxes[:,0],
                   all_boxes[:,1],
                   all_boxes[:,2],
                   all_boxes[:,3],
                   all_boxes[:,4]
                  ])

        boxes = boxes.T
        # boxes = [x1, y1, x2, y2, score, reg] reg= [px1, py1, px2, py2] (in prediction)
        align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
        align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
        align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
        align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh

        # refine the boxes
        boxes_align = np.vstack([ align_topx,
                              align_topy,
                              align_bottomx,
                              align_bottomy,
                              all_boxes[:, 4]
                              ])
        boxes_align = boxes_align.T

        #remove invalid box
        valindex = [True for _ in range(boxes_align.shape[0])]   
        for i in range(boxes_align.shape[0]):
            if boxes_align[i][2]-boxes_align[i][0]<=3 or boxes_align[i][3]-boxes_align[i][1]<=3:
                valindex[i]=False
                print('pnet has one smaller than 3')
            else:
                if boxes_align[i][2]<1 or boxes_align[i][0]>w-2 or boxes_align[i][3]<1 or boxes_align[i][1]>h-2:
                    valindex[i]=False
                    print('pnet has one out')
        boxes_align=boxes_align[valindex,:]
        boxes = boxes[valindex,:]
        return boxes, boxes_align

    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """
        # im: an input image
        im = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
        h, w, c = im.shape

        if dets is None:
            return None,None
        if dets.shape[0]==0:
            return None, None
        detss = dets
        # return square boxes
        dets = convert_to_square(dets)
        detsss = dets
        # rounds
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        # cropped_ims_tensors = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
        cropped_ims_tensors = []
        for i in range(num_boxes):
            try:
                tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
                tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = im[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            except:    
                print(dy[i],edy[i],dx[i],edx[i],y[i],ey[i],x[i],ex[i],tmpw[i],tmph[i])
                print(dets[i])
                print(detss[i])
                print(detsss[i])
                print(h,w)
                exit()
            crop_im = cv2.resize(tmp, (24, 24))
            crop_im_tensor = convert_image_to_tensor(crop_im)
            # cropped_ims_tensors[i, :, :, :] = crop_im_tensor
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = torch.stack(cropped_ims_tensors).to(device)
       

        cls_map, reg = self.rnet_detector(feed_imgs)
        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()
        # landmark = landmark.cpu().data.numpy()
        
        keep_inds = np.where(cls_map > self.thresh[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
            # landmark = landmark[keep_inds]
        else:
            return None, None
        keep = nms(boxes, 0.5)

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        # keep_landmark = landmark[keep]


        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1


        boxes = np.vstack([ keep_boxes[:,0],
                              keep_boxes[:,1],
                              keep_boxes[:,2],
                              keep_boxes[:,3],
                              keep_cls[:,0]
                            ])

        align_topx = keep_boxes[:,0] + keep_reg[:,0] * bw
        align_topy = keep_boxes[:,1] + keep_reg[:,1] * bh
        align_bottomx = keep_boxes[:,2] + keep_reg[:,2] * bw
        align_bottomy = keep_boxes[:,3] + keep_reg[:,3] * bh

        boxes_align = np.vstack([align_topx,
                               align_topy,
                               align_bottomx,
                               align_bottomy,
                               keep_cls[:, 0]
                             ])

        boxes = boxes.T
        boxes_align = boxes_align.T

        #remove invalid box
        valindex = [True for _ in range(boxes_align.shape[0])]   
        for i in range(boxes_align.shape[0]):
            if boxes_align[i][2]-boxes_align[i][0]<=3 or boxes_align[i][3]-boxes_align[i][1]<=3:
                valindex[i]=False
                print('rnet has one smaller than 3')
            else:
                if boxes_align[i][2]<1 or boxes_align[i][0]>w-2 or boxes_align[i][3]<1 or boxes_align[i][1]>h-2:
                    valindex[i]=False
                    print('rnet has one out')
        boxes_align=boxes_align[valindex,:]
        boxes = boxes[valindex,:]
        return boxes, boxes_align

    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes_align: numpy array
            boxes after calibration
        landmarks_align: numpy array
            landmarks after calibration

        """
        im = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR) 
        h, w, c = im.shape

        if dets is None:
            return None, None
        if dets.shape[0]==0:
            return None, None

        detss = dets
        dets = convert_to_square(dets)
        
        
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]


        # cropped_ims_tensors = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
        cropped_ims_tensors = []
        for i in range(num_boxes):
            try:
                tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
                # crop input image
                tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            except:
                print(dy[i],edy[i],dx[i],edx[i],y[i],ey[i],x[i],ex[i],tmpw[i],tmph[i])
                print(dets[i])
                print(detss[i])
                print(h,w)
            crop_im = cv2.resize(tmp, (48, 48))
            crop_im_tensor = convert_image_to_tensor(crop_im)
            # cropped_ims_tensors[i, :, :, :] = crop_im_tensor
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = torch.stack(cropped_ims_tensors)
        feed_imgs = feed_imgs.to(device)

        cls_map, reg, landmark = self.onet_detector(feed_imgs)

        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()
        landmark = landmark.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.thresh[2])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None
        #更改一下，改小了会少出现些重框！1!
        keep = nms(boxes, 0.1, mode="Union")

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        keep_landmark = landmark[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1


        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        align_landmark_topx = keep_boxes[:, 0]
        align_landmark_topy = keep_boxes[:, 1]

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0]
                                 ])

        boxes_align = boxes_align.T

        landmark =  np.vstack([
                                 align_landmark_topx + keep_landmark[:, 0] * bw,
                                 align_landmark_topy + keep_landmark[:, 1] * bh,
                                 align_landmark_topx + keep_landmark[:, 2] * bw,
                                 align_landmark_topy + keep_landmark[:, 3] * bh,
                                 align_landmark_topx + keep_landmark[:, 4] * bw,
                                 align_landmark_topy + keep_landmark[:, 5] * bh,
                                 align_landmark_topx + keep_landmark[:, 6] * bw,
                                 align_landmark_topy + keep_landmark[:, 7] * bh,
                                 align_landmark_topx + keep_landmark[:, 8] * bw,
                                 align_landmark_topy + keep_landmark[:, 9] * bh,
                                 ])

        landmark_align = landmark.T
        
        return boxes_align, landmark_align


    def detect_face(self,img):
        """Detect face over image
        """
        boxes_align = np.array([])
        landmark_align =np.array([])

#         t = time.time()

        # pnet
        if self.pnet_detector:
            boxes, boxes_align = self.detect_pnet(img)
            if boxes_align is None:
                return np.array([]), np.array([])

#             t1 = time.time() - t
#             t = time.time()

        # rnet
        if self.rnet_detector:
            boxes, boxes_align = self.detect_rnet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

#             t2 = time.time() - t
#             t = time.time()

        # onet
        if self.onet_detector:
            boxes_align, landmark_align = self.detect_onet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

#             t3 = time.time() - t
#             t = time.time()
#             print("time cost " + '{:.3f}'.format(t1+t2+t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3))

        return boxes_align, landmark_align


# In[ ]:


def vis_face(im_array, dets, landmarks, save_name):
    """Visualize detection results before and after calibration

    Parameters:
    ----------
    im_array: numpy.ndarray, shape(1, c, h, w)
        test image in rgb
    dets1: numpy.ndarray([[x1 y1 x2 y2 score]])
        detection results before calibration
    
    thresh: float
        boxes with scores > thresh will be drawn in red otherwise yellow

    Returns:
    -------
    """
    fig, ax = plt.subplots(figsize=(10,10),dpi=100)
   
    ax.imshow(im_array)

    for i in range(dets.shape[0]):
        bbox = dets[i, :4]            
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor='yellow', linewidth=0.9)
        ax.add_patch(rect)

    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            landmarks_one = landmarks[i, :]
            landmarks_one = landmarks_one.reshape((5, 2))
            for j in range(5):
                cir1 = Circle(xy=(landmarks_one[j, 0], landmarks_one[j, 1]), radius=10, alpha=0.4, color="red")
                ax.add_patch(cir1)
    plt.axis("off")
    fig.savefig(save_name,pad_inches=0.0)
    fig.show()

