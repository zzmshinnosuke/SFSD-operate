#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2024-01-13 15:08:36
# @Author: zzm

import json, os, glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import numpy as np

class Sketch():
    def __init__(self, sketchPath=None, sketch=None):
        if sketchPath is not None:
            self.path = sketchPath
            self.sketch = self.loadSketch()
        elif sketch is not None:
            self.sketch = sketch
        else:
            self.sketch = None
        assert self.sketch, "Please give the path to the sketch or the loaded sketch！"
    
    def loadSketch(self):
        with open(self.path, "r") as f:
            try:
                load_dict = json.load(f)
                return  load_dict
            except json.decoder.JSONDecodeError:
                print(self.path)
        return None
    
    def get_resolution(self):
        return self.sketch["resolution"]
    
    def get_objects(self):
        return self.sketch["objects"]
    
    def get_strokes(self):
        strokes = []
        for obj in self.sketch["objects"]:
            strokes.extend(obj["strokes"])
        return strokes
    
    def get_object_num(self):
        return len(self.get_objects())
    
    def get_stroke_num(self):
        return len(self.get_strokes())
    
    def get_all_category(self):
        all_category=[obj["category"] for obj in self.get_objects()]
        return list(set(all_category))
    
    def gen_image_MPL(self, stroke_width=1.5):
        # Visualize the sketch
        def color2hex(color):
            new_color = "#"
            for co in color:
                if co>=16:
                    new_color+=str(hex(co))[-2:]
                elif co==0:
                    new_color += "00"
                else:
                    new_color += "0"+str(hex(co))[-1:]
            new_color += str(hex(255))[-2:]
            return new_color
        width, height = self.sketch["resolution"]
        plt.figure(figsize=(width/96, height/96), dpi=96)
        x = []
        y = [] 
        color = color2hex([0,0,0])
        for obj in self.sketch["objects"]:
            for stroke in obj["strokes"]: 
                points = stroke["points"]
                for point in points:
                    x.append(point[0])
                    y.append(point[1])
                nodes = np.array([x, y])
                x1 = nodes[0]
                y1 = nodes[1]
                plt.plot(x1, y1, color=color, linewidth=stroke_width, linestyle="-") 
                x.clear()  
                y.clear()
        plt.axis('off')
        ax = plt.gca()  # 获取坐标轴信息
        ax.invert_yaxis()  # y轴反向
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        img = np.array(canvas.renderer.buffer_rgba())
        plt.clf()
        plt.close("all")
        return img
    
    def gen_image_PIL(self, stroke_width=1.5):
        strokes = self.get_strokes()
        width, height = self.get_resolution()
        src_img = Image.new("RGB", (width, height), (255,255,255))
        draw = ImageDraw.Draw(src_img)
        for stroke in strokes:
            points = tuple(tuple(p) for p in stroke['points'])
            draw.line(points, fill=(0,0,0), width=stroke_width)
        return np.array(src_img)
    
    def save_image(self, image_path=None, image_name=None, mode="MPL", stroke_width=1.5):
        if mode == "MPL":
            image = self.gen_image_MPL(stroke_width)
        else:
            image = self.gen_image_PIL(stroke_width)
        im = Image.fromarray(image)
        if not image_name:
            if self.path:
                image_name = os.path.basename(self.path).split(".")[0].strip()
            else:
                image_name = "sketchImg"
        if not image_path:
            image_path = os.getcwd()
        # print(image_name, image_path)
        im.save(os.path.join(image_path, '{}.png'.format(image_name)))

class SFSD():
    def __init__(self, root_path, split="train") -> None:
        self.root_path = root_path
        self.split = split
        self.sketch_names = self.get_sketches()
        self.sketches = self.load_sketches()
        self.CATEGORIES = self.load_category()

    def get_sketches(self):
        sketch_names = []
        assert self.split in ['train', 'test', 'traintest'], 'unknown split {}'.format(self.split)
        if self.split=='traintest':
            paths = glob.glob(os.path.join(self.root_path, 'sketches', '*.json'))
            sketch_names = [os.path.basename(path).split('.')[0] for path in paths]
        else:
            filename_txt = 'train_names.txt' if self.split=='train' else 'test_names.txt'
            filename_txt = os.path.join(self.root_path, filename_txt)
            assert os.path.exists(filename_txt), 'not find {}'.format(filename_txt)
            with open(filename_txt, 'r') as f:
                sketch_names = [line.strip() for line in f.readlines()]
                
        assert len(sketch_names)>0, 'no json file find in {}'.format(self.root_path)
        return sketch_names

    def load_sketches(self):
        print("load sketch json:::::::::::::::::::::::::::::::::::")
        sketches = list()
        for sketch_name in tqdm(self.sketch_names):
            sketch = Sketch(sketchPath = os.path.join(self.root_path, 'sketches', sketch_name+'.json'))
            sketch.name = sketch_name
            sketches.append(sketch)
        return sketches
    
    def load_category(self):
        filename = "categories_info.json"
        category_path = os.path.join(self.root_path, filename)
        assert os.path.exists(category_path), "category file {} not exist.".format(filename)
        with open(category_path, 'r') as fp:
            obj_names = json.load(fp)
        id_cat = dict()
        for cat in obj_names:
            id = obj_names[cat]["id"]
            id_cat[id] = {"category":cat, "color":obj_names[cat]["color"]}    
        CATEGORIES = dict()
        CATEGORIES['cat2id'] = obj_names
        CATEGORIES['id2cat'] = id_cat
        return CATEGORIES

if __name__ == '__main__':
#     sketch = Sketch(sketchPath="/home/zzm/datasets/SFSD-open/sketches/000000239559.json")
#     print(sketch.get_object_num())
#     sketch.save_image(image_name="MPL", stroke_width=2, mode="MPL")
    sfsd = SFSD(root_path="/home/zzm/datasets/SFSD-open", split="traintest")
    print(sfsd.CATEGORIES)
