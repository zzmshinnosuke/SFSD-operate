# 将SFSD转成SceneSketcherv2所用的格式。

from SFSD import Sketch, SFSD

import os, math
from tqdm import tqdm
import numpy as np

background = ["playground", "snowfield", "tree", "grass", "river", "cloud", "stone", "mountain", "road", "boundary", "fence", "house", "others"]
foreground = ['horse', 'truck', 'giraffe', 'car', 'baseball bat', 'motorcycle', 'bear', 'zebra', 'backpack', 'baseball glove', 'kite', 'elephant', 'cow', 'skis', 'surfboard', 'airplane', 'bird', 'sheep', 'snowboard', 'dog', 'person', 'sports ball', 'skateboard', 'bicycle', 'bus', 'frisbee', 'tennis racket']

def gen_foreground(CATEGORIES):
    cats = CATEGORIES["cat2id"].keys()
    foreground = set(cats) - set(background)
    print(foreground)

def get_bbox(obj):
    all_points = []
    for stroke in obj["strokes"]: 
        all_points.extend(stroke["points"])    
    min_xy=np.min(np.array(all_points),0).tolist()
    max_xy=np.max(np.array(all_points),0).tolist()
    min_xy.extend(max_xy)
    return [int(i) for i in min_xy]

def gen_graph(sketch):
    # features
    #id,className,bboxleft,bboxtop,bboxright,bboxbottom
    # 0,12,110,159,293,310
    # edge
    # id1,id2,weight
    # 0,1,152.2005256232711
    feats = []
    feats.append(["id","className","bboxleft","bboxtop","bboxright","bboxbottom"])

    edges = []
    edges.append(["id1","id2","weight"])

    obj_id = 0
    for obj in sketch.get_objects():
        if obj["category"] in foreground:
            cat_id = foreground.index(obj["category"])
            bbox = get_bbox(obj)
            feat = [obj_id, cat_id]
            feat.extend(bbox)
            feats.append(feat)
            obj_id += 1
    for i in range(1, len(feats)):
        for j in range(i+1, len(feats)):
            p1_cx = (feats[i][2] + feats[i][4])/2
            p1_cy = (feats[i][3] + feats[i][5])/2
            p2_cx = (feats[j][2] + feats[j][4])/2
            p2_cy = (feats[j][3] + feats[j][5])/2
            bbox_dis = math.sqrt(math.pow((p2_cx - p1_cx), 2) + math.pow((p2_cy - p1_cy), 2))
            edges.append([feats[i][0], feats[j][0], bbox_dis])
    return feats, edges

def save_csv(path, filename, content):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), 'w') as fe:
        for line in content:
            tmp = [str(c) for c in line]
            fe.write(','.join(tmp) + '\n')

if __name__ == '__main__':
    SFSD_path = '/home/zzm/datasets/SFSD-open'
    target_path = '/home/zzm/SFSD-lf/'
    for split in ["train", "test"]:
        sfsd = SFSD(root_path=SFSD_path, split=split)
        # gen_foreground(sfsd.CATEGORIES)
        for sketch in tqdm(sfsd.sketches):
            feats, edges = gen_graph(sketch)
            save_csv(os.path.join(target_path, split, "GraphFeatures"), sketch.name+".csv", feats)
            save_csv(os.path.join(target_path, split, "GraphEdges"), sketch.name+".csv", edges)
            image_path = os.path.join(target_path, split, "Image")
            os.makedirs(image_path, exist_ok=True)
            sketch.save_image(image_path=image_path, image_name=sketch.name, mode="PIL", stroke_width=1)
    