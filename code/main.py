from tools.gen_data import generate_new_masks
import glob
from PIL import Image
from PIL import ImageOps
import numpy as np
import cv2
import string
import random
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # root_img = "data/SFR_Data/"
    # _ = generate_new_masks(root_img,
    #                            "data/output/",
    #                            1000, False)
    # # cases = generate_new_masks(root_img, "data/output/", 100, False)
    # " python test.py --dataroot ./datasets/facades/testB/ --name facades_pix2pix --model test --netG unet_256 --direction BtoA --dataset_mode single --norm batch"
    root_img = "data/results/final_masks/*"
    save_file = "data/submit_files/"
    ki = 0
    for mask_file in glob.glob(root_img):
        mask = Image.open(mask_file)
        mask = ImageOps.grayscale(mask)
        mask = np.array(mask)
        mask = (mask > 65).astype('uint8')
        # mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, -1)
        current_id = mask_file.split("/")[-1].split(".")[0]

        print(f"mask {ki}: {current_id}")
        new_id = '61'+ ''.join(random.choices(string.ascii_lowercase + string.digits, k=22))

        # Portale
        img = Image.open("data/results/SFR_portale/"+current_id+"_fake.png")
        img = ImageOps.grayscale(img)
        img = np.array(img)
        # img = cv2.resize(img, (512, 512))
        img = np.expand_dims(img, -1)
        full_img = np.concatenate((img, mask), axis=-1)
        with open(save_file + new_id +"-"+'T1 phase portale'+'.npy', 'wb') as f_portale:
            np.save(f_portale, full_img)
            print(f"saved {new_id +'-'+'T1 phase portale'+'.npy'}")

        # Centre
        img = Image.open("data/results/SFR_centre/"+current_id+"_fake.png")
        img = ImageOps.grayscale(img)
        img = np.array(img)
        # img = cv2.resize(img, (512, 512))
        img = np.expand_dims(img, -1)
        full_img = np.concatenate((img, mask), axis=-1)
        with open(save_file + new_id +"-"+'T2 centrée'+'.npy', 'wb') as f_centre:
            np.save(f_centre, full_img)
            print(f"saved {new_id + '-' + 'T2 centrée' + '.npy'}")

        # arterialle
        img = Image.open("data/results/SFR_arterielle/"+current_id+"_fake.png")
        img = ImageOps.grayscale(img)
        img = np.array(img)
        # img = cv2.resize(img, (512, 512))
        img = np.expand_dims(img, -1)
        full_img = np.concatenate((img, mask), axis=-1)
        with open(save_file + new_id +"-"+'T1 phase artérielle'+'.npy', 'wb') as f_arterialle:
            np.save(f_arterialle, full_img)
            print(f"saved {new_id + '-' + 'T1 phase artérielle' + '.npy'}")
        print("==========================================\n")