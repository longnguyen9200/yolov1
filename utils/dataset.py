import torch
# import matplotlib.pyplot as plt
import cv2

class TrafficSign(torch.utils.data.Dataset):
    def __init__(self, directory, path, s=7, b=2, c=4, transform=None):
        self.s = s
        self.c = c
        self.b = b
        self.transform = transform

        self.img_path = []
        self.img_info = []

        f = open(path)
        lines = [line for line in f.readlines()]

        for idx in range(len(lines)):
            self.img_path.append(lines[idx].replace('/home/my_name/', directory).split('\n')[0])
            self.img_info.append(lines[idx].replace('/home/my_name/', directory).split('\n')[0].replace('.jpg', '.txt'))
    
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img_info = self.img_info[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = []
        with open(img_info) as f:
            for line in f.readlines():
                line = line.replace('\n', '').split()
                l, x, y, w, h = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
                boxes.append([x, y, w, h, l])

        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=boxes)
            transformed_img = transformed['image']
            transformed_bboxes = transformed['bboxes']

            transformed_bboxes = torch.tensor(transformed_bboxes)
        else:
            boxes = torch.tensor(boxes)

        label_matrix = torch.zeros((self.s, self.s, self.b*5 + self.c))
        for box in transformed_bboxes:
            x, y, w, h, class_label = box[0], box[1], box[2], box[3], box[4]
            class_label = int(class_label)

            loc = [self.s * x, self.s * y]
            i, j = int(loc[1]), int(loc[0])
            y = loc[1] - i
            x = loc[0] - j 
            w, h = (
                w * self.s,
                h * self.s,
            )

            if label_matrix[i, j, 4] == 0:
                label_matrix[i, j ,4] = 1

                box_coordinates = torch.tensor([x, y, w, h])

                label_matrix[i, j, 5:9] = box_coordinates

                label_matrix[i, j, class_label] = 1
        
        return transformed_img, label_matrix
            