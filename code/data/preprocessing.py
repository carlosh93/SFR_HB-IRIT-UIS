import shutil
import os
import glob

if __name__ == '__main__':
    os.system("tar -xvf data_1_2021.tar")
    os.system("tar -xvf data_2_2021.tar")
    os.system("tar -xvf data_3_2021.tar")

    for f in os.listdir('export_d_1'):
        shutil.move('export_d_1/' + f, 'export_2021_jfr/' + f)
    os.system("rmdir export_d_1")
    shutil.move('export_2021_jfr', 'SFR_Data')
    root_data = 'SFR_Data'
    data_list = glob.glob(root_data + "/*.npy")
    data_list.sort()
    for idx, img in enumerate(data_list):
        patient_id = data_list[idx].split("/")[-1].split("_")[0]
        print(patient_id)
        if not os.path.exists(os.path.join(root_data, patient_id)):
            os.makedirs(os.path.join(root_data, patient_id), exist_ok=True)
        shutil.move(img, os.path.join(root_data, patient_id, data_list[idx].split("/")[-1]))

    os.system(f"rm {root_data+'/*.meta'}")

    for folder in glob.glob(root_data+"/*"):
        if len(os.listdir(folder)) != 3:
            os.system(f"rm -R {folder}")