import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import shutil
from lxml import etree
from PIL import Image
import torchvision.transforms as T


def create_mask_file(width, height, background, shapes):
    mask = np.full((height, width, 3), background, dtype=np.uint8)
    for shape in shapes:
        points = [tuple(map(float, p.split(','))) for p in shape['points'].split(';')]
        points = np.array([(int(p[0]), int(p[1])) for p in points])
        points = points.astype(int)
        mask = cv2.drawContours(mask, [points], -1, color=(0, 0, 255), thickness=5)
        mask = cv2.fillPoly(mask, [points], color=(0, 0, 255))
    return mask


def get_liver_mask(new_size, name_id, modality):
    ann_folder = 'tools/liver_segments'
    if modality == "T2 centrée":
        cvat_xml = os.path.join(ann_folder, 'liver_seg_centre.xml')
        name_id += '_2.png'
    if modality == "T1 phase artérielle":
        cvat_xml = os.path.join(ann_folder, 'liver_seg_arterielle.xml')
        name_id += '_3.png'
    if modality == "T1 phase portale":
        cvat_xml = os.path.join(ann_folder, 'liver_seg_portale.xml')
        name_id += '_1.png'
    root = etree.parse(cvat_xml).getroot()
    anno = []

    image_name_attr = ".//image[@name='{}']".format(name_id)

    for image_tag in root.iterfind(image_name_attr):
        image = {}
        for key, value in image_tag.items():
            image[key] = value
        image['shapes'] = []
        for poly_tag in image_tag.iter('polygon'):
            polygon = {'type': 'polygon'}
            for key, value in poly_tag.items():
                polygon[key] = value
            image['shapes'].append(polygon)
        for box_tag in image_tag.iter('box'):
            box = {'type': 'box'}
            for key, value in box_tag.items():
                box[key] = value
            box['points'] = "{0},{1};{2},{1};{2},{3};{0},{3}".format(
                box['xtl'], box['ytl'], box['xbr'], box['ybr'])
            image['shapes'].append(box)
        image['shapes'].sort(key=lambda x: int(x.get('z_order', 0)))
        anno.append(image)

    mask = np.zeros((int(anno[0]['width']), int(anno[0]['height']), 3), np.uint8)
    for image in anno:
        mask = create_mask_file(int(anno[0]['width']),
                                int(anno[0]['height']),
                                mask,
                                image['shapes'])

    mask = cv2.resize(mask, (new_size, new_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask[..., -1].astype('float32') / 255.0).astype('uint8')
    return mask


def new_mask(old_mask, liver_mask):
    old_mask = Image.fromarray(old_mask)
    affine_transformer = T.RandomAffine(degrees=(0, 45), translate=(0.3, 0.3), scale=(0.2, 1.5))

    mask = affine_transformer(old_mask)
    j_mask = 2 * liver_mask + np.array(mask).astype('uint8')
    new_mask = j_mask.copy()
    new_mask[new_mask == 1] = 0
    new_mask[new_mask == 2] = 0
    new_mask[new_mask == 3] = 1
    kernel = np.ones((2, 2), np.uint8)
    return cv2.erode(new_mask, kernel, iterations=1), j_mask


def skip_img(img_name):
    black_list = [
        "6160709a2b487ccb98fbc5a8"
    ]
    return img_name in black_list


def random_name():
    import random
    import string

    # printing lowercase
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(10))
    #
    # # printing uppercase
    # letters = string.ascii_uppercase
    # print(''.join(random.choice(letters) for i in range(10)))
    #
    # # printing letters
    # letters = string.ascii_letters
    # print(''.join(random.choice(letters) for i in range(10)))
    #
    # # printing digits
    # letters = string.digits
    # print(''.join(random.choice(letters) for i in range(10)))
    #
    # # printing punctuation
    # letters = string.punctuation
    # print(''.join(random.choice(letters) for i in range(10)))


def generate_new_masks(root_dir, dest_dir, num_cases, plot_over_seg):
    """
    root_img: folder with images
    dest_dir: folder to save masks
    num_cases: number of masks to generate = num_cases*3
    """
    cases = []
    data_list = glob.glob(root_dir + "/*")
    data_list.sort()
    idx = 0
    while len(cases) < num_cases:
        if idx >= len(data_list):
            idx = 0
        imgs_files = glob.glob(data_list[idx] + "/*.npy")
        if len(imgs_files) > 0:
            if skip_img(imgs_files[0].split("/")[-2]):
                idx += 1
                continue
        out_dict = {}
        for img_file in imgs_files:
            img = np.load(img_file)
            ori_mask = img[..., 1].astype('float32')
            ori_mask = cv2.resize(ori_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            patient_id = img_file.split("/")[-2]
            modality = img_file.split("/")[-1].split("_")[-1].strip(".npy")
            liver_mask = get_liver_mask(256, patient_id, modality)
            assert np.sum(np.unique(liver_mask)) != 0
            mask, over_img = new_mask(ori_mask, liver_mask)
            while np.sum(np.unique(mask)) == 0 or len(np.where(mask > 0)[0]) < 81:
                # No tumor inside the liver -> repeat the previous procedure
                mask, over_img = new_mask(ori_mask, liver_mask)
            over_rgb = np.zeros((256, 256, 3))
            over_rgb[..., 0][over_img == 1] = 255
            over_rgb[..., 1][over_img == 2] = 255
            over_rgb[..., 2][over_img == 3] = 255
            img = img[..., 0].astype('float32')
            img = cv2.resize(img, (256, 256))
            img -= img.min()
            img /= img.max()
            img *= 255
            img = img.astype('uint8')
            lt = np.random.randint(10, 100)
            edges = cv2.Canny(img, lt, 240).astype('float32') / 255.0
            edges = edges * (-1 * mask + 1)
            join_mask = np.zeros((256, 256, 3))
            join_mask[..., 0] = edges * 255
            join_mask[..., 2] = mask * 255
            join_mask = join_mask.astype('uint8')
            rname = random_name()
            if modality == "T1 phase artérielle":
                out_dict['arterielle'] = join_mask
                dest_dir_art = os.path.join(dest_dir, "arterielle")
                if not os.path.exists(dest_dir_art):
                    os.makedirs(dest_dir_art, exist_ok=True)
                if plot_over_seg:
                    if not os.path.exists(os.path.join(dest_dir_art, 'over_segm')):
                        os.makedirs(os.path.join(dest_dir_art, 'over_segm'), exist_ok=True)
                    cv2.imwrite(os.path.join(dest_dir_art, 'over_segm', patient_id + "_over_3_"+rname+".jpg"), over_rgb)
                cv2.imwrite(os.path.join(dest_dir_art, patient_id + "_joint_3_"+rname+".jpg"), join_mask)
            if modality == "T1 phase portale":
                out_dict['portale'] = join_mask
                dest_dir_port = os.path.join(dest_dir, "portale")
                if not os.path.exists(dest_dir_port):
                    os.makedirs(dest_dir_port, exist_ok=True)
                if plot_over_seg:
                    if not os.path.exists(os.path.join(dest_dir_port, 'over_segm')):
                        os.makedirs(os.path.join(dest_dir_port, 'over_segm'), exist_ok=True)
                    cv2.imwrite(os.path.join(dest_dir_port, 'over_segm', patient_id + "_over_1_"+rname+".jpg"), over_rgb)
                cv2.imwrite(os.path.join(dest_dir_port, patient_id + "_joint_1_"+rname+".jpg"), join_mask)
            if modality == "T2 centrée":
                out_dict['centrée'] = join_mask
                dest_dir_centre = os.path.join(dest_dir, "centree")
                if not os.path.exists(dest_dir_centre):
                    os.makedirs(dest_dir_centre, exist_ok=True)
                if plot_over_seg:
                    if not os.path.exists(os.path.join(dest_dir_centre, 'over_segm')):
                        os.makedirs(os.path.join(dest_dir_centre, 'over_segm'), exist_ok=True)
                cv2.imwrite(os.path.join(dest_dir_centre, 'over_segm', patient_id + "_over_2_"+rname+".jpg"), over_rgb)
                cv2.imwrite(os.path.join(dest_dir_centre, patient_id + "_joint_2_"+rname+".jpg"), join_mask)
        cases.append(out_dict)
        idx += 1

    return cases