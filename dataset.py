import numpy as np
import os, csv, random, gc, pickle
import nibabel as nib
from scipy.ndimage.interpolation import zoom
from scipy.misc import imresize 

"""
In seg file
--------------
Label 1: necrotic and non-enhancing tumor
Label 2: edema 
Label 4: enhancing tumor
Label 0: background

MRI
-------
whole/complete tumor: 1 2 4
core: 1 4
enhance: 4
"""
###============================= SETTINGS ===================================###
DATA_SIZE = 'half' # (small, half or all)

save_dir = "data/train_dev_all/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

HGG_data_path = "data/Training"
survival_csv_path = "data/survival_data.csv"
###==========================================================================###

survival_id_list = []
survival_age_list =[]
survival_peroid_list = []

with open(survival_csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for idx, content in enumerate(reader):
        survival_id_list.append(content[0])
        survival_age_list.append(float(content[1]))
        survival_peroid_list.append(float(content[2]))

print(len(survival_id_list)) #163

if DATA_SIZE == 'all':
    HGG_path_list = list(map(os.path.abspath,os.listdir(path = HGG_data_path)))
elif DATA_SIZE == 'half':
    HGG_path_list = list(map(os.path.abspath,os.listdir(path = HGG_data_path)))[0:13]# DEBUG WITH SMALL DATA
 
elif DATA_SIZE == 'small':
    HGG_path_list = list(map(os.path.abspath,os.listdir(path = HGG_data_path)))[0:6] # DEBUG WITH SMALL DATA
else:
    exit("Unknown DATA_SIZE")
print(len(HGG_path_list)) #210 #75

HGG_name_list = [os.path.basename(p) for p in HGG_path_list]


survival_id_from_HGG = []
for i in survival_id_list:
        survival_id_from_HGG.append(i)
 

print(len(survival_id_from_HGG)) #163, 0


# use 42 from 210 (in 163 subset) and 15 from 75 as 0.8/0.2 train/dev split

# use 126/42/42 from 210 (in 163 subset) and 45/15/15 from 75 as 0.6/0.2/0.2 train/dev/test split
index_HGG = list(range(0, len(survival_id_from_HGG)))

# random.shuffle(index_HGG)
# random.shuffle(index_HGG)

if DATA_SIZE == 'all':
    dev_index_HGG = index_HGG[-7:-4]
    test_index_HGG = index_HGG[-4:]
    tr_index_HGG = index_HGG[:-7]
elif DATA_SIZE == 'half':
    dev_index_HGG = index_HGG[11:13]  # DEBUG WITH SMALL DATA
    test_index_HGG = index_HGG[13:15]
    tr_index_HGG = index_HGG[:12]
elif DATA_SIZE == 'small':
    dev_index_HGG = index_HGG[4:6]   # DEBUG WITH SMALL DATA
    # print(index_HGG, dev_index_HGG)
    # exit()
    test_index_HGG = index_HGG[5:6]
    tr_index_HGG = index_HGG[0:4]


survival_id_dev_HGG = [survival_id_from_HGG[i] for i in dev_index_HGG]
survival_id_test_HGG = [survival_id_from_HGG[i] for i in test_index_HGG]
survival_id_tr_HGG = [survival_id_from_HGG[i] for i in tr_index_HGG]


survival_age_dev = [survival_age_list[survival_id_list.index(i)] for i in survival_id_dev_HGG]
survival_age_test = [survival_age_list[survival_id_list.index(i)] for i in survival_id_test_HGG]
survival_age_tr = [survival_age_list[survival_id_list.index(i)] for i in survival_id_tr_HGG]

survival_period_dev = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_dev_HGG]
survival_period_test = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_test_HGG]
survival_period_tr = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_tr_HGG]


data_types = ['DWI','Flair', 'T1', 'T2']
data_types_mean_std_dict = {i: {'mean': 0.0, 'std': 1.0} for i in data_types}


# calculate mean and std for all data types

# preserving_ratio = 0.0
# preserving_ratio = 0.01 # 0.118 removed
# preserving_ratio = 0.05 # 0.213 removed
# preserving_ratio = 0.10 # 0.359 removed

#==================== LOAD ALL IMAGES' PATH AND COMPUTE MEAN/ STD
import matplotlib.pyplot as plt
for i in data_types:
    data_temp_list = []
    for j in HGG_name_list:
        img_path = os.path.join(HGG_data_path, j, j + '_' + i + '.nii')
	#print(img_path)
        img = nib.load(img_path).get_data()
        img1=np.zeros((240,240,153), np.float32)
        for c in range(1, 153):
            X=img[:, :, c]
            X=imresize(X, (240, 240), mode='F')
            img1[:, :, c]=X
    data_temp_list.append(img1)
          #print("a")

    data_temp_list = np.asarray(data_temp_list)
    print(data_temp_list.shape)
    m = np.mean(data_temp_list)
    s = np.std(data_temp_list)
    data_types_mean_std_dict[i]['mean'] = m
    data_types_mean_std_dict[i]['std'] = s
del data_temp_list
print(data_types_mean_std_dict)

with open(save_dir + 'mean_std_dict.pickle', 'wb') as f:
    pickle.dump(data_types_mean_std_dict, f, protocol=4)


##==================== GET NORMALIZE IMAGES
X_train_input = []
X_train_target = []
# X_train_target_whole = [] # 1 2 4
# X_train_target_core = [] # 1 4
# X_train_target_enhance = [] # 4

X_dev_input = []
X_dev_target = []
# X_dev_target_whole = [] # 1 2 4
# X_dev_target_core = [] # 1 4
# X_dev_target_enhance = [] # 4

print(" HGG Validation")

for i in survival_id_dev_HGG:
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(HGG_data_path, i, i + '_' + j + '.nii')
        img = nib.load(img_path).get_data()
        img1=np.zeros((240,240,153), np.float32)
        for c in range(1, 153):
            X=img[:, :, c]
            X=imresize(X, (240, 240), mode='F')
            img1[:, :, c]=X
        img1 = (img1 - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img1 = img1.astype(np.float32)
        all_3d_data.append(img1)

    seg_path = os.path.join(HGG_data_path, i, i + 'OT.nii')
    seg_img = nib.load(seg_path).get_data()
    seg_img1=np.zeros((240,240,153), np.float32)
    for c in range(1, 153):
        X=seg_img[:, :, c]
        X=imresize(X, (240, 240), mode='F')
        seg_img1[:, :, c]=X
    seg_img1 = np.transpose(seg_img1, (1, 0, 2))


    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]), axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
        combined_array.astype(np.float32)
        X_dev_input.append(combined_array)

        seg_2d = seg_img1[:, :, j]
        # whole = np.zeros_like(seg_2d)
        # core = np.zeros_like(seg_2d)
        # enhance = np.zeros_like(seg_2d)
        # for index, x in np.ndenumerate(seg_2d):
        #     if x == 1:
        #         whole[index] = 1
        #         core[index] = 1
        #     if x == 2:
        #         whole[index] = 1
        #     if x == 4:
        #         whole[index] = 1
        #         core[index] = 1
        #         enhance[index] = 1
        # X_dev_target_whole.append(whole)
        # X_dev_target_core.append(core)
        # X_dev_target_enhance.append(enhance)
        seg_2d.astype(int)
        X_dev_target.append(seg_2d)
    del all_3d_data
    gc.collect()
    print("finished {}".format(i))



X_dev_input = np.asarray(X_dev_input, dtype=np.float32)
X_dev_target = np.asarray(X_dev_target)#, dtype=np.float32)
# print(X_dev_input.shape)
# print(X_dev_target.shape)

# with open(save_dir + 'dev_input.pickle', 'wb') as f:
#     pickle.dump(X_dev_input, f, protocol=4)
# with open(save_dir + 'dev_target.pickle', 'wb') as f:
#     pickle.dump(X_dev_target, f, protocol=4)

# del X_dev_input, X_dev_target

print(" HGG Train")
for i in survival_id_tr_HGG:
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(HGG_data_path, i, i + '_' + j + '.nii')
        img = nib.load(img_path).get_data()
        img1=np.zeros((240,240,153), np.float32)
        for c in range(1, 153):
            X=img[:, :, c]
            X=imresize(X, (240, 240), mode='F')
            img1[:, :, c]=X
        img1 = (img1 - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img1 = img1.astype(np.float32)
        all_3d_data.append(img1)

    seg_path = os.path.join(HGG_data_path, i, i + 'OT.nii')
    seg_img = nib.load(seg_path).get_data()
    seg_img1=np.zeros((240,240,153), np.float32)
    for c in range(1, 153):
        X=seg_img[:, :, c]
        X=imresize(X, (240, 240), mode='F')
        seg_img1[:, :, c]=X
    seg_img1 = np.transpose(seg_img1, (1, 0, 2))
    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]), axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
        combined_array.astype(np.float32)
        X_train_input.append(combined_array)

        seg_2d = seg_img1[:, :, j]
        # whole = np.zeros_like(seg_2d)
        # core = np.zeros_like(seg_2d)
        # enhance = np.zeros_like(seg_2d)
        # for index, x in np.ndenumerate(seg_2d):
        #     if x == 1:
        #         whole[index] = 1
        #         core[index] = 1
        #     if x == 2:
        #         whole[index] = 1
        #     if x == 4:
        #         whole[index] = 1
        #         core[index] = 1
        #         enhance[index] = 1
        # X_train_target_whole.append(whole)
        # X_train_target_core.append(core)
        # X_train_target_enhance.append(enhance)
        seg_2d.astype(int)
        X_train_target.append(seg_2d)
    del all_3d_data
    print("finished {}".format(i))
    # print(len(X_train_target))



X_train_input = np.asarray(X_train_input, dtype=np.float32)
X_train_target = np.asarray(X_train_target)#, dtype=np.float32)
print(X_train_input.shape)
print(X_train_target.shape)

# with open(save_dir + 'train_input.pickle', 'wb') as f:
#     pickle.dump(X_train_input, f, protocol=4)
# with open(save_dir + 'train_target.pickle', 'wb') as f:
#     pickle.dump(X_train_target, f, protocol=4)



# X_train_target_whole = np.asarray(X_train_target_whole)
# X_train_target_core = np.asarray(X_train_target_core)
# X_train_target_enhance = np.asarray(X_train_target_enhance)


# X_dev_target_whole = np.asarray(X_dev_target_whole)
# X_dev_target_core = np.asarray(X_dev_target_core)
# X_dev_target_enhance = np.asarray(X_dev_target_enhance)


# print(X_train_target_whole.shape)
# print(X_train_target_core.shape)
# print(X_train_target_enhance.shape)

# print(X_dev_target_whole.shape)
# print(X_dev_target_core.shape)
# print(X_dev_target_enhance.shape)



# with open(save_dir + 'train_target_whole.pickle', 'wb') as f:
#     pickle.dump(X_train_target_whole, f, protocol=4)

# with open(save_dir + 'train_target_core.pickle', 'wb') as f:
#     pickle.dump(X_train_target_core, f, protocol=4)

# with open(save_dir + 'train_target_enhance.pickle', 'wb') as f:
#     pickle.dump(X_train_target_enhance, f, protocol=4)

# with open(save_dir + 'dev_target_whole.pickle', 'wb') as f:
#     pickle.dump(X_dev_target_whole, f, protocol=4)

# with open(save_dir + 'dev_target_core.pickle', 'wb') as f:
#     pickle.dump(X_dev_target_core, f, protocol=4)

# with open(save_dir + 'dev_target_enhance.pickle', 'wb') as f:
#     pickle.dump(X_dev_target_enhance, f, protocol=4)


### TEST ###

X_test_input = []
X_test_target = []


print("HGG Test")

for i in survival_id_test_HGG:
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(HGG_data_path, i, i + '_' + j + '.nii')
        img = nib.load(img_path).get_data()
        img1=np.zeros((240,240,153), np.float32)
        X=np.zeros((240,240), np.float32)
        for c in range(1, 153):
            X=img[:, :, c]
            X=imresize(X, (240, 240), mode='F')
            img1[:, :, c]=X;
        img1 = (img1 - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img1 = img1.astype(np.float32)
        all_3d_data.append(img1)

    seg_path = os.path.join(HGG_data_path, i, i + 'OT.nii')
    seg_img = nib.load(seg_path).get_data()
    seg_img1=np.zeros((240,240,153), np.float32)
    X=np.zeros((240,240), np.float32)
    for c in range(1, 153):
        X=seg_img[:, :, c]
        X=imresize(X, (240, 240), mode='F')
        seg_img1[:, :, c]=X;
    seg_img1 = np.transpose(seg_img1, (1, 0, 2))


    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]), axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
        combined_array.astype(np.float32)
        X_test_input.append(combined_array)

        seg_2d = seg_img1[:, :, j]

        seg_2d.astype(int)
        X_test_target.append(seg_2d)
    del all_3d_data
    gc.collect()
    print("finished {}".format(i))



X_test_input = np.asarray(X_test_input, dtype=np.float32)
X_test_target = np.asarray(X_test_target)#, dtype=np.float32)
print(X_test_input.shape)
print(X_test_target.shape)