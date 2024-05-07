class SoccerNetDataset(dataset.Dataset):
    def __init__(self, root='data', train=True):
        self.train = train

        if self.train:
            self.root = os.path.join(root, 'train')
            self.image_set = 'train'
        else:
            self.root = os.path.join(root, 'test')
            self.image_set = 'val'

        self.folder2id = dict(zip(os.listdir(self.root), range(len(os.listdir(self.root)))))

        self.images = []
        self.annotations = []

        self.id2label = {1: "player", 2: "goalkeeper", 3: "referee", 4: "ball", 5: "unknown"}
        self.label2id = {"player": 1, "goalkeeper": 2, "referee": 3, "ball": 4, "unknown": 5}

        #this is where (mx,my,h,w) of COCO format changes to (mx,my,Mx,My)
        self.prepare = ConvertCocoPolysToMask(return_masks=False)

        #for data augmentation, this is where (mx, my, Mx, My) changes to (cx, cy, h, w) via normalize method in (official detr)
        self._transform = make_coco_transforms(self.image_set)

        for folder in os.listdir(self.root):
            #read gameinfo.ini file
            gameinfo = configparser.ConfigParser()
            gameinfo.read_file(open(os.path.join(self.root, folder, 'gameinfo.ini'), 'r'))

            #make a dictionary for tracklet_id to label
            tracklet2label = {}
            for section in gameinfo.sections():
                for k,v in gameinfo.items(section):
                    if k.find('trackletid') != -1:
                        if v.find('player') != -1:
                            tracklet2label[k[11:]] = 'player'
                        elif v.find('ball') != -1:
                            tracklet2label[k[11:]] = 'ball'
                        elif v.find('referee') != -1:
                            tracklet2label[k[11:]] = 'referee'
                        elif v.find('goalkeeper') != -1:
                            tracklet2label[k[11:]] = 'goalkeeper'
                        elif v.find('other') != -1:
                            tracklet2label[k[11:]] = 'unknown'

            #local image and annotation list
            im = []
            ann = []

            for file in sorted(os.listdir(os.path.join(self.root, folder, 'img1'))):
                im.append(os.path.join(self.root, folder, 'img1', file))
                ann.append([])

            with open(os.path.join(self.root,folder,'gt/gt.txt'), 'r') as f:
                for line in f:
                    line = line.rstrip().split(',')

                    if int(line[4]) < 5 or int(line[5]) < 5:
                        continue
                    else:
                        ann[int(line[0])-1].append({'isCrowd': 0,
                                                'category_id': self.label2id[tracklet2label[line[1]]],
                                                'bbox': (int(line[2]), int(line[3]), int(line[4]), int(line[5])),
                                                'area': int(line[4])*int(line[5]),
                                                'image_id': self.folder2id[folder]*750+int(line[0])-1})
            self.images += im
            self.annotations += ann

        for annotation in self.annotations:
            for instance in annotation:
                bbox = instance['bbox']
                if bbox[2] < 5 or bbox[3] < 5:
                    raise ValueError("bounding box cannot have height or width less than 5")


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(self.images[idx])
        annotation = self.annotations[idx]
        image_id = idx

        target = {'image_id': image_id, 'annotations': annotation}
        image, target = self.prepare(image, target)
        if self._transform is not None:
            image, target = self._transform(image, target)
        return image, target
