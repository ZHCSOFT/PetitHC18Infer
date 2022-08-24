import os
import re
import sys
import time
from glob import glob
from functools import cmp_to_key

import cv2
import numpy as np

from openvino.inference_engine import IECore


model_dir = './models'
pending_dir = './pending'
save_dir = './results'

norm_mean_b = 0.485
norm_std_b = 0.229
norm_mean_g = 0.456
norm_std_g = 0.224
norm_mean_r = 0.406
norm_std_r = 0.225

colors_set = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
              (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0),
              (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
              (192, 0, 128), (64, 128, 128), (192, 128, 128),
              (0, 64, 0), (128, 64, 0), (0, 192, 0),
              (128, 192, 0), (0, 64, 128),
              (128, 64, 12)]


class MaskProc:
    def _sortByContourArea(self, contour1, contour2):
        # Descending order
        if cv2.contourArea(contour1) > cv2.contourArea(contour2):
            return -1
        elif cv2.contourArea(contour1) < cv2.contourArea(contour2):
            return 1
        return 0

    def _imfill(self, input_img: np.ndarray)-> np.ndarray:
        """
        :param input_img: Only accept single gray or binary image
        """
        ret, bin_img = cv2.threshold(input_img, thresh=128, maxval=255, type=cv2.THRESH_BINARY)
        img_floodfill = bin_img.copy()
        height, width = bin_img.shape[:2]
        mask = np.zeros((height+2, width+2), np.uint8)
        cv2.floodFill(img_floodfill, mask=mask, seedPoint=(0, 0), newVal=(255))
        img_floodfill_invert = cv2.bitwise_not(img_floodfill)
        
        return cv2.bitwise_or(bin_img, img_floodfill_invert)

    def _getLargestArea(self, inputMask: np.ndarray, largestNum: int=2)-> tuple([np.ndarray, tuple]):
        contours, hierarchy = cv2.findContours(image=inputMask.copy(),
                                               mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
        contours = sorted(list(contours), key=cmp_to_key(self._sortByContourArea))
        blankImg = np.zeros(inputMask.shape).astype('uint8')
        largeContours = []
        if len(contours) < largestNum:
            largestNum = len(contours)
            print('len(contours) is smaller than largest _num, value =', len(contours))
        for i in range (0, largestNum):
            contour = contours[i]
            largeContours.append(contour)
        # fill the small contours with pixel 0
        resultMask = cv2.drawContours(blankImg, largeContours, contourIdx=-1, color=[255], thickness=cv2.FILLED)

        return resultMask, largeContours

    def getEllipseByMask(self, inputMask: np.ndarray, maxCount: int=1):
        contours, hierarchy = cv2.findContours(image=inputMask.copy(),
                                               mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
        all_points = []
        contours = sorted(list(contours), key=cmp_to_key(self._sortByContourArea))
        numCount = 0
        for contour in contours:
            if numCount >= maxCount:
                break
            numCount += 1
            for point in contour:
                all_points.append(point)
        all_points = np.asarray(all_points)
        convex_result = cv2.convexHull(all_points)
        ellipse = cv2.fitEllipse(convex_result)
        # (center_X, center_Y), (r_short, r_long), angle = ellipse
        # angle: vertical_line-y-+ -> r_long
        return ellipse

    def getSkullThicknessByMask(self, inputMask:np.ndarray):
        _, largestSkullContours = self._getLargestArea(inputMask, largestNum=2)
        meanResult = 0
        for largestSkullContour in largestSkullContours:
            contourArea = cv2.contourArea(largestSkullContour)
            contourLength = cv2.arcLength(largestSkullContour, closed=True)
            meanResult += contourArea / contourLength * 2
        meanResult = meanResult / len(largestSkullContours)

        return meanResult

    def drawEllipseOnBlank(self, imgShape:tuple, ellipse:tuple, fill=True)-> np.ndarray:
        blankImg = np.zeros(imgShape).astype('uint8')
        if fill:
            return self._imfill(cv2.ellipse(blankImg, ellipse, (255)))
        else:
            return cv2.ellipse(blankImg, ellipse, (255))

    def drawEllipseAxis(self,
                        originImage:np.ndarray, ellipse:tuple, minusThickness:float=0.,
                        drawLong=False, drawShort=True):

        if originImage.shape[2] == 1:
            input_bgr_image = cv2.cvtColor(originImage, cv2.COLOR_GRAY2BGR)
        else:
            input_bgr_image = originImage

        (center_X, center_Y), (r2_short, r2_long), angle = ellipse
        # for short axis
        rad_angle = np.radians(angle-90)
        
        x_bias_short = r2_short / 2 * np.sin(rad_angle)
        y_bias_short = r2_short / 2 * np.cos(rad_angle)
        x_thickness_short = minusThickness * np.sin(rad_angle)
        y_thickness_short = minusThickness * np.cos(rad_angle)

        result_img = input_bgr_image

        if drawShort:
            if rad_angle > np.pi/2:
                short_endpoint_a = (int(np.round(center_X + (x_bias_short - x_thickness_short))),
                                    int(np.round(center_Y + (y_bias_short - y_thickness_short))))
                short_endpoint_b = (int(np.round(center_X - x_bias_short)), int(np.round(center_Y - y_bias_short)))
            
            # elif rad_angle > np.pi/4:
            #     short_endpoint_a = (int(center_X + (x_bias_short - x_thickness_short)),
            #                         int(center_Y - (y_bias_short - y_thickness_short)))
            #     short_endpoint_b = (int(center_X - x_bias_short), int(center_Y + y_bias_short))
            else:
                short_endpoint_a = (int(np.round(center_X + (x_bias_short - x_thickness_short))),
                                    int(np.round(center_Y - (y_bias_short - y_thickness_short))))
                short_endpoint_b = (int(np.round(center_X - x_bias_short)), int(np.round(center_Y + y_bias_short)))
            result_img = cv2.line(result_img, short_endpoint_a, short_endpoint_b, (255, 0, 0), 5)
        if drawLong:
        # for long axis
            x_bias_long = r2_long / 2 * np.cos(rad_angle)
            y_bias_long = r2_long / 2 * np.sin(rad_angle)
            x_thickness_long = minusThickness * np.cos(rad_angle)
            y_thickness_long = minusThickness * np.sin(rad_angle)
            long_endpoint_a = (int(np.round(center_X + x_bias_long - x_thickness_long)),
                               int(np.round(center_Y + y_bias_long - y_thickness_long)))
            long_endpoint_b = (int(np.round(center_X - x_bias_long)), int(np.round(center_Y - y_bias_long)))
            result_img = cv2.line(result_img, long_endpoint_a, long_endpoint_b, (0, 255, 0), 5)

        return result_img


class ModelProc:
    def __init__(self, modelPath, featureSize: tuple[int, int]):
        self.modelPath = modelPath
        self.featureSize = featureSize
        self.ie = IECore()
        self.net = self.ie.read_network(model=modelPath)
        self.net.batch_size = 1
        self.execNet = self.ie.load_network(network = self.net, device_name='CPU')

        self.inputBlob = next(iter(self.net.input_info))
        self.outputHCBlob = 'modelOutput'
        self.outputSkullBlob = '1159'

    def getPredMasksByImage(self, inputImg: np.ndarray):
        feat_H = self.featureSize[0]; feat_W = self.featureSize[1]
        inputImg = inputImg.astype('float32')
        inputImg = cv2.resize(inputImg, (feat_W, feat_H))

        imgB = inputImg[:, :, 0]; imgG = inputImg[:, :, 1]; imgR = inputImg[:, :, 2]
        imgB = (imgB / 255 - norm_mean_b) / norm_std_b
        imgG = (imgG / 255 - norm_mean_g) / norm_std_g
        imgR = (imgR / 255 - norm_mean_r) / norm_std_r
        inputImg = cv2.merge([imgB, imgG, imgR])

        inputImg = inputImg.transpose((2, 0, 1))
        inputImg = np.expand_dims(inputImg, axis=0)

        results = self.execNet.infer(inputs={self.inputBlob: inputImg})

        resultHC = np.asarray(results[self.outputHCBlob]).reshape([2, feat_W, feat_H]).transpose((1, 2, 0))
        resultHC = resultHC.argmax(axis=-1)
        resultHC = resultHC.astype('uint8')
        resultSkull = np.asarray(results[self.outputSkullBlob]).reshape([2, feat_W, feat_H]).transpose((1, 2, 0))
        resultSkull = resultSkull.argmax(axis=-1)
        resultSkull = resultSkull.astype('uint8')
        return resultHC, resultSkull


if __name__ == '__main__':
    modelName = 'DeeplabV3Plus_Siamese_resnet'
    pendingFiles = glob(pending_dir + '/*', recursive=True)

    pendingFiles = [x for x in pendingFiles if 'README.md' not in x]
    assert not len(pendingFiles) == 0

    m_Exec = ModelProc(model_dir + '/' + modelName + '.onnx', (513, 513))
    m_maskProc = MaskProc()

    save_path = save_dir + '/' + str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))

    assert not os.path.exists(save_path)
    
    os.mkdir(save_path); os.mkdir(save_path+'/mask_HC'); os.mkdir(save_path+'/mask_Skull')
    os.mkdir(save_path+'/visualize')

    for pendingFile in pendingFiles:
        pendingImg = cv2.imread(pendingFile, cv2.IMREAD_COLOR)
        predHCMask, predSkullMask = m_Exec.getPredMasksByImage(pendingImg)

        predHCMask = cv2.resize(predHCMask, (pendingImg.shape[1], pendingImg.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        predSkullMask = cv2.resize(predSkullMask, (pendingImg.shape[1], pendingImg.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(save_path+'/mask_HC/'+os.path.basename(pendingFile), predHCMask*255)
        cv2.imwrite(save_path+'/mask_Skull/'+os.path.basename(pendingFile), predSkullMask*255)

        ellipse = m_maskProc.getEllipseByMask(predHCMask)
        (center_X, center_Y), (r_short, r_long), angle = ellipse

        predHCEllipseMask = m_maskProc.drawEllipseOnBlank(predHCMask.shape, ellipse)
        skullThickness = m_maskProc.getSkullThicknessByMask(predSkullMask)
        axisResult = m_maskProc.drawEllipseAxis(pendingImg, ellipse, skullThickness)

        cv2.imwrite(save_path+'/visualize/'+os.path.basename(pendingFile), axisResult)
