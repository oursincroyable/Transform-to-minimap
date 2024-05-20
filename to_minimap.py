from PIL import Image
import cv2 as cv
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

import torch
import torchvision.transforms as transforms

class To_Minimap():
    '''
    field_image_path : path to worldcup_field_model.png,
    path_DPT : path to a trained DPT model for football field registration,
    path_YOLOS : path to a trained YOLOS model for players detection.
    '''
    def __init__(self,field_image_path = 'KpSFR/assets/worldcup_field_model.png',
                 path_DPT='KpSFR/dataset/0409DPT8model.pt',
                 path_YOLOS='yolos10model.pt'):
        #width, height of football field in yards
        #ground-truth homography induces a transformation from an image of size 1280*720 to that of 114.83*74.37
        self.field_width = 114.83
        self.field_height = 74.37
        self.grid_field = self._form_grid(self.field_width, self.field_height, 13, 7)

        #load a minimap image of size 1050*680 downloaded from https://github.com/ericsujw/KpSFR
        self.field = cv.imread(field_image_path)
        self.field_green, self.field_gray = self._to_green_gray(self.field)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #load a trained DPT model
        self.DPT = torch.load(path_DPT, map_location=self.device)
        self.DPT.eval()

        #labelling data for detection
        self.id2label = {1: "player", 2: "goalkeeper", 3: "referee", 4: "ball", 5: "unknown"}
        self.label2id = {"player": 1, "goalkeeper": 2, "referee": 3, "ball": 4, "unknown": 5}
        self.label2color = {'player':(255,255,255),
               'goalkeeper':(255,255,0),
               'referee':(51,255,51),
               'ball':(255,0,0),
               'unknown':(0,0,0)}

        #load pretrained YOLOS model
        self.YOLOS = torch.load(path_YOLOS, map_location=self.device)
        self.YOLOS.eval()

    #make an equidistributed keypoints on an ideal minimap
    def _form_grid(self, field_width, field_height, dw, dh):
        dw = np.linspace(0, field_width, dw)
        dh = np.linspace(0, field_height, dh)
        grid_dw, grid_dh = np.meshgrid(dw, dh, indexing='ij')
        return np.stack((grid_dw, grid_dh), axis=2).reshape(-1,2)

    #convert the minimap image to green and gray minimap images
    def _to_green_gray(self, field):
        green = Image.new(mode='RGB', size=(1050, 680), color=(0, 154, 23))
        gray = Image.new(mode='RGB', size=(1050, 680), color=(120, 120, 120))

        return cv.bitwise_or(np.array(green), np.array(field)), cv.bitwise_or(np.array(gray), np.array(field))

    # preprocess an image to an input of DPT
    def _preprocess_DPT(self, image):
        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        image = transforms.Resize((480, 480), interpolation=transforms.InterpolationMode.BICUBIC)(image)
        return image.unsqueeze(dim=0)

    # run DPT model to obtain keypoints detection
    def _predict_keypoints(self, image):

        input = self._preprocess_DPT(image)
        input.to(self.device)

        #model inference
        with torch.no_grad():
            outputs = self.DPT(input)

        #postprocess to return segmentation mask
        logits = outputs.logits
        resized_logits = torch.nn.functional.interpolate(
            logits[0].unsqueeze(dim=0), size=(720, 1280), mode="bilinear", align_corners=False
        )
        pred_mask = resized_logits[0].argmax(dim=0)

        return pred_mask.cpu().numpy()

    #predict homography from segmentation mask
    def _predict_homography(self, mask):
        classes = np.unique(mask)

        if len(classes) < 4:
            raise ValueError("Number of predicted keypoints is less than four!")

        classes = classes[classes != 0]

        src = []
        tgt = []

        for c in classes:
            src.append(ndimage.center_of_mass(mask == c)[::-1])
            tgt.append(self.grid_field[c - 1])

        return cv.findHomography(np.array(src).reshape(-1, 1, 2),
                                 np.array(tgt).reshape(-1, 1, 2),
                                 method=cv.RANSAC,
                                 ransacReprojThreshold=8)[0]

    #make a minimap image from a homography matrix
    def _to_minimap(self, hom):
        # generate a mask to make minimap image
        mask = np.array(Image.new(mode='RGB', size=(115, 74), color=(255, 255, 255)))
        # image
        mask = cv.warpPerspective(mask, np.linalg.inv(hom), dsize=(1280, 720))
        # inverse image
        mask = cv.warpPerspective(mask, np.array([[1050 / 115, 0, 0], [0, 680 / 74, 0], [0, 0, 1]]) @ hom,
                                  dsize=(1050, 680))

        image = cv.bitwise_or(cv.bitwise_and(self.field_green, mask), cv.bitwise_and(self.field_gray, cv.bitwise_not(mask)))

        return image

    #makes a minimap image from a football broadcast image
    def to_minimap(self, image):
        return self._to_minimap(self._predict_homography(self._predict_keypoints(image)))


    #preprocess images for YOLOS model inputs
    def _preprocess_YOLOS(self, image):
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        return [image.to(self.device)]

    #make a coordinate change from cxcywh to minx, miny, maxx, maxy
    def _box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    #postprocess outputs of YOLOS
    def _postprocess_YOLOS(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob = torch.nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = self._box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

    #return coordinates on image and representing colors of detected objects
    def _to_coordinates(self, image):

        input = self._preprocess_YOLOS(image)

        with torch.no_grad():
            outputs = self.YOLOS(input)

        target_sizes = torch.tensor((720, 1280))
        results = self._postprocess_YOLOS(outputs, target_sizes.unsqueeze(dim=0))[0]
        scores, labels, boxes = results['scores'], results['labels'], results['boxes']

        image = np.array(image)

        coord = []
        color = []

        image_players = []
        coord_players = []

        for score, label, (xmin, ymin, xmax, ymax) in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
            if score > 0.9:
                if self.id2label[label] != 'player':
                    coord.append([(xmin + xmax) / 2, ymax, 1])
                    color.append(self.label2color[self.id2label[label]])
                else:
                    image_players.append(
                        np.mean(np.mean(image[int(ymin):int(ymax), int(xmin):int(xmax)], axis=0), axis=0))
                    coord_players.append([(xmin + xmax) / 2, ymax, 1])
                    
        #apply k-means algorithm for classify players into two teams based on colors 
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(image_players)
        label_players = kmeans.labels_

        color_players = []
        for l in label_players:
            if l:
                color_players.append([255, 255, 255])
            else:
                color_players.append([0, 0, 255])

        coord += coord_players
        color += color_players

        return coord, color
    
    def to_minimap_with_detection(self, image):

        coord, color = self._to_coordinates(image)
        hom = self._predict_homography(self._predict_keypoints(image))
        minimap = self._to_minimap(hom)

        proj_coord = np.array(coord).reshape(-1, 3) @ (
                    np.array([[1050 / 115, 0, 0], [0, 680 / 74, 0], [0, 0, 1]]) @ hom).T
        proj_coord /= proj_coord[:, 2, np.newaxis]
        proj_coord = proj_coord[:, :2]

        radius = 13
        for c, col in zip(proj_coord, color):
            cv.circle(minimap, (int(c[0]), int(c[1])), radius, col, -1)

        return minimap
