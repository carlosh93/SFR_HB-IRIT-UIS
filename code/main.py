from tools.gen_data import generate_new_masks


if __name__ == '__main__':
    root_img = "data/SFR_Data/"
    cases = generate_new_masks(root_img, "data/output/", 100, False)
    " python test.py --dataroot ./datasets/facades/testB/ --name facades_pix2pix --model test --netG unet_256 --direction BtoA --dataset_mode single --norm batch"