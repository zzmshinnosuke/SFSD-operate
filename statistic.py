from SFSD import Sketch

import argparse
import os
from tqdm import tqdm
import numpy as np

def sta_sketch_category_num(sketches, interval=10):
    '''
    统计每个草图包含物体最多、最少、平均类别数
    '''
    print("统计每个草图包含的物体类别数:")
    cat_nums = list()
    res = dict()
    for sketch in tqdm(sketches):
        cat_num = len(sketch.get_all_category())
        cat_nums.append(cat_num)
        
    print('max seq_len is {},min seq_len is {},mean seq_len is {}'.format(max(cat_nums),min(cat_nums),np.mean(cat_nums)))

def sta_category_sketch_num(sketches, interval=1000):
    '''
    统计每个类别物体出现的最多、最少、平均草图数
    '''
    print("统计每个类别物体出现的草图数:")
    
    sketch_nums = list()
    cats = list()
    for sketch in tqdm(sketches):
        cat = sketch.get_all_category()
        cats.extend(cat)
    
    cats_set = set(cats)
    for cat in cats_set:
        sketch_nums.append(cats.count(cat))
    
    print('max seq_len is {}, min seq_len is {}, mean seq_len is {}'.format(max(sketch_nums), min(sketch_nums), np.mean(sketch_nums)))

def sta_sketch_point_num(sketches, interval=1000):
    '''
    统计每个草图包含的点数
    '''
    print("统计每个草图包含的点数:")
    
    point_lens = list()
    res = dict()
    for sketch in tqdm(sketches):
        point_len = sketch.get_points_len()
        point_lens.append(point_len)
        div=point_len//interval
        temp_name = (div+1)*interval
        if temp_name in res.keys():
            res[temp_name] += 1 
        else :
            res[temp_name] = 1 
        # if(point_len>17000):
        #     print(sketch.sketch_name, point_len)
    
    sorted_res = sorted(res.items(), key=lambda item: item[0], reverse=False) 
    print('max seq_len is {},min seq_len is {},mean seq_len is {}'.format(max(point_lens),min(point_lens),np.mean(point_lens)))
    print(sorted_res)
    print(dict(sorted_res).keys(),dict(sorted_res).values())

def sta_sketch_stroke_num(sketches, interval=100):
    '''
    统计每个草图包含的笔画数
    '''
    print("统计每个草图包含的笔画数:")
    stroke_lens = list()
    res = dict()
    for sketch in tqdm(sketches):
        stroke_len = sketch.get_stroke_num()
        stroke_lens.append(stroke_len)
        div = stroke_len//interval
        temp_name = (div+1)*interval
        if temp_name in res.keys():
            res[temp_name] += 1 
        else :
            res[temp_name]=1  
        if stroke_len > 700:
             print(sketch.sketch_name, stroke_len)
            
    sorted_res = sorted(res.items(), key=lambda item: item[0], reverse=False)    
    print('all stroke_len is {},max stroke_len is {},min stroke_len is {},mean stroke_len is {}'.format(sum(stroke_lens),max(stroke_lens),min(stroke_lens),np.mean(stroke_lens)))
    print(sorted_res)
    print(dict(sorted_res).keys(),dict(sorted_res).values())
    
def sta_sketch_object_num(sketches):
    '''
    统计每个草图中的物体数
    '''
    print("统计每个草图包含的物体数:")
    object_lens=list()
    res=dict()
    for sketch in tqdm(sketches):
        object_len = sketch.get_object_num()
        object_lens.append(object_len)
        temp_name = object_len
        if temp_name in res.keys():
            res[temp_name] += 1 
        else :
            res[temp_name] = 1 
        if object_len == 1:
            print(sketch.sketch_name)
    
    sorted_res = sorted(res.items(), key=lambda item: item[0], reverse=False)
    print('all objects_len is {}, max objects_len is {}, min objects_len is {},  mean objects_len is {}'.format(sum(object_lens),max(object_lens),min(object_lens),np.mean(object_lens)))
    print(sorted_res)
    print(dict(sorted_res).keys(), dict(sorted_res).values())
    
def sta_stroke_point_num(sketches,interval=100):
    '''
    统计每个笔画包含的点数
    '''
    print("统计每个笔画的采样点数:")
    point_lens = list()
    res = dict()
    for sketch in tqdm(sketches):
        strokes = sketch.get_strokes()
        for stroke in strokes:
            point_len = stroke.get_points_len()
            point_lens.append(point_len)
            div=point_len//interval
            temp_name = (div+1)*interval
            if temp_name in res.keys():
                res[temp_name] += 1 
            else :
                res[temp_name] = 1  
    sorted_res = sorted(res.items(), key=lambda item: item[0], reverse=False)
    print('max stroke_point_len is {},min stroke_point_len is {}, std stroke_point_len is {}, mean stroke_point_len is {}'.format(max(point_lens),min(point_lens),np.std(point_lens),np.mean(point_lens)))
    print(sorted_res)
    print(dict(sorted_res).keys(),dict(sorted_res).values())
    
def sta_stroke_length_num(sketches,interval=100):
    '''
    统计笔画不同长度个数
    '''
    print("统计笔画不同长度个数:")
    stroke_lens = list()
    res = dict()
    for sketch in tqdm(sketches):
        strokes = sketch.get_strokes()
        for stroke in strokes:
            stroke_len = stroke.get_stroke_len()
            stroke_lens.append(stroke_len)
            div = int(stroke_len)//interval
            temp_name = (div+1)*interval
            if temp_name in res.keys():
                res[temp_name] += 1 
            else :
                res[temp_name] = 1 
    
    sorted_res = sorted(res.items(), key=lambda item: item[0], reverse=False)
    print('max stroke_length_len is {},min stroke_length_len is {}, mean stroke_length_len is {}'.format(max(stroke_lens),min(stroke_lens),np.mean(stroke_lens)))
    print(sorted_res)
    print(dict(sorted_res).keys(),dict(sorted_res).values())
        
def sta_object_num_of_every_cat(sketches):
    '''
    统计每个类别物体的个数
    '''
    cat_num = dict()
    all_nums = 0
    for sketch in tqdm(sketches):
        for item in sketch.get_items():
            all_nums += 1
            if item.category in cat_num:
                cat_num[item.category] += 1
            else: 
                cat_num[item.category] = 1
                
            if item.category == "suitcase":
                print(sketch.sketch_name)
#     cat_num=sorted(cat_num.items(), key = lambda item:item[1], reverse = True)
#     less100_cats = list()
#     num = 0
#     for te in cat_num:
#         if te[1] < 100:
#             num += 1
#             less100_cats.append(te[0])
#     print("统计每个类别物体的个数:")
#     print(len(cat_num), cat_num, num, all_nums)
    
    cat_by_name_origin = sorted(cat_num.items(), key = lambda item:item[0], reverse = False)
    cat_by_name = dict(cat_by_name_origin)
    others_num = cat_by_name.pop("others")
    cat_name_nums = cat_by_name.values()
    cat_name_nums = list(cat_name_nums)
    cat_name_nums.append(others_num) 
    
    weights = np.array(cat_name_nums)
    weights = np.median(weights) / weights
    weights = np.around(weights, decimals = 2) #保留两位小数
    
    print("统计每个类别的笔画数:")
    print(len(cat_by_name_origin), cat_by_name_origin)
    weights = [str(w) for w in weights] # 方便直接复制过去使用
    print("每个类别权值为：{}, 总共：{}".format((',').join(weights), len(weights)))
    
    return cat_num

def sta_stroke_num_of_every_cat(sketches, interval=1000):
    '''
    统计每个类别的笔画数
    '''
    print("统计每个类别物体出现的草图数:")
    
    cat_num = dict()
    all_nums = 0
    for sketch in tqdm(sketches):
        for item in sketch.get_items():
            all_nums += item.get_strokes_len()
            if item.category in cat_num:
                cat_num[item.category] += item.get_strokes_len()
            else:
                cat_num[item.category] = item.get_strokes_len()
    #按照名称顺序排序后，在将others类别的值放在最后，与类别定义的顺序保持一致，训练的时候可以直接给每个类别指定权值。
    cat_by_name_origin = sorted(cat_num.items(), key = lambda item:item[0], reverse = False)
    cat_by_name = dict(cat_by_name_origin)
    others_num = cat_by_name.pop("others")
    cat_name_nums = cat_by_name.values()
    cat_name_nums = list(cat_name_nums)
    cat_name_nums.append(others_num) 
    
    weights = np.array(cat_name_nums)
    weights = np.median(weights) / weights
    weights = np.around(weights, decimals = 2) #保留两位小数
    
    print("统计每个类别的笔画数:")
    print(len(cat_num), cat_by_name_origin, all_nums)
    weights = [str(w) for w in weights] # 方便直接复制过去使用
    print("每个类别权值为：{}, 总共：{}".format((',').join(weights), len(weights)))
    return cat_num

def sta_width_height(sketches):
    widths = list()
    heights = list()
    for sketch in tqdm(sketches):
        width, height = sketch.resolution
        widths.append(width)
        heights.append(height)
    print("max width is {}, min width is {}. max height is {}, min height is {}".format(np.max(widths), np.min(widths), np.max(heights), np.min(heights)))

def change_sketch_catogery(sketches, less100_cats):
    for sketch in tqdm(sketches):
        sketch.save_new_json('~/tmp/sketch2',less100_cats)    
    
def get_all_sketches(path, sketches_json):
    print("load sketch json:")
    sketches = list()
    for sketch_json in tqdm(sketches_json):
        sketch = Sketch(sketchPath = os.path.join(path, sketch_json))
        sketch.name = os.path.basename(sketch_json).split('.')[0].strip()
        sketches.append(sketch)
    return sketches
     
def get_parser(prog='statistics sketch'):
    parser=argparse.ArgumentParser(prog)

    parser.add_argument('--sketch_path',
                        default='~/datasets/SFSD-open',
                        required=True,
                        help='the path of sketch')
    parser.add_argument('--sketch_cat_num',
                    type=bool,
                    default=False,
                    help='statistic the number of object category in every sketch')
    parser.add_argument('--cat_sketch_num',
                    type=bool,
                    default=False,
                    help='statistic the number of sketch in every category')
    parser.add_argument('--sketch_point_num',
                        type=bool,
                        default=False,
                        help='statistic the number of points in every sketch')
    parser.add_argument('--sketch_stroke_num',
                        type=bool,
                        default=False,
                        help='statistic the number of strokes in every sketch')
    parser.add_argument('--sketch_object_num',
                        type=bool,
                        default=False,
                        help='statistic the number of objects in every sketch')
    parser.add_argument('--stroke_point_num',
                        type=bool,
                        default=False,
                        help='statistic the number of points in every stroke')
    parser.add_argument('--stroke_length_num',
                        type=bool,
                        default=False,
                        help='statistic the number of every stroke length')
    parser.add_argument('--cat_object_num',
                        type=bool,
                        default=False,
                        help='statistic the number of objects in every cat')
    parser.add_argument('--cat_stroke_num',
                    type=bool,
                    default=False,
                    help='statistic the number of stroke in every category')
    parser.add_argument('--sta_width_height',
                        type=bool,
                        default=False,
                        help='statistic the width and height of sketches')
    parser.add_argument('--interval',
                        type=int,
                        default=100,
                        help='the interval to statistics')
    
    return parser.parse_args()

# python scripts/statistics.py --sketch_path ~/datasets/SFSD/sketch --sketch_point_num True --interval 1000

if __name__ == '__main__':
    args=get_parser()
    sketches_json=[file for file in os.listdir(args.sketch_path) if os.path.isfile(os.path.join(args.sketch_path, file))]
    sketches=get_all_sketches(args.sketch_path, sketches_json)
    print("the number of sketches is {}".format(len(sketches)))
    if args.sketch_stroke_num:
        sta_sketch_stroke_num(sketches,args.interval)
    if args.sketch_object_num:
        sta_sketch_object_num(sketches)
    # if args.stroke_length_num:
    #     sta_stroke_length_num(sketches,args.interval)
#     if args.cat_object_num:
#         sta_object_num_of_every_cat(sketches)
#     if args.cat_stroke_num:
#         sta_stroke_num_of_every_cat(sketches)
#     if args.sketch_cat_num:
#         sta_sketch_category_num(sketches,args.interval)
#     if args.cat_sketch_num:
#         sta_category_sketch_num(sketches,args.interval)
# #     change_sketch_catogery(sketches,less100_cats)
#     if args.sta_width_height:
#         sta_width_height(sketches)

# python statistic.py --sketch_path ~/datasets/SFSD-open/sketches --sketch_stroke_num True