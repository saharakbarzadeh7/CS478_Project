
import pandas as pd
import numpy as np
import os
import nibabel as nib

file_path = 'D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/name_mapping_pre-processing.csv'
df = pd.read_csv(file_path)

labels = np.zeros(357)
start = np.zeros(357)
end = np.zeros(357)
inputs = np.zeros((357, 160, 210, 100, 4))

rootdir = 'D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/'
subdirs = os.listdir(rootdir)
i = 0

for sub in subdirs:
    if sub in df["BraTS_2020_subject_ID"].values:
        print(sub)
        patient_slices = []
        t1_file_path = rootdir + '/' + sub + '/' + sub + '_t1.nii.gz'
        t1ce_file_path = rootdir + '/' + sub + '/' + sub + '_t1ce.nii.gz'
        t2_file_path = rootdir + '/' + sub + '/' + sub + '_t2.nii.gz'
        flair_file_path = rootdir + '/' + sub + '/' + sub + '_flair.nii.gz'

        start[i] = df.loc[df['BraTS_2020_subject_ID'] == sub, 'start'].values[0]
        end[i] = df.loc[df['BraTS_2020_subject_ID'] == sub, 'end'].values[0]

        grade = df.loc[df['BraTS_2020_subject_ID'] == sub, 'Grade'].values[0]
        print(grade)
        if grade == 'HGG':
            labels[i] = 1
        else:
            labels[i] = 0

        if os.path.exists(t1_file_path):
            nii_img = nib.load(t1_file_path)
            t1 = nii_img.get_fdata()
            inputs[i, :, :, :, 0] = t1[40:200, 20:230, 27:127]

            nii_img = nib.load(t1ce_file_path)
            t1ce = nii_img.get_fdata()
            inputs[i, :, :, :, 1] = t1ce[40:200, 20:230, 27:127]

            nii_img = nib.load(t2_file_path)
            t2 = nii_img.get_fdata()
            inputs[i, :, :, :, 2] = t2[40:200, 20:230, 27:127]

            nii_img = nib.load(flair_file_path)
            flair = nii_img.get_fdata()
            inputs[i, :, :, :, 3] = flair[40:200, 20:230, 27:127]
            i += 1

print(inputs.shape)
print(labels.shape)
print(start.shape)
print(end.shape)

# Saving inputs and labels
np.save('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/inputs.npy', inputs)
np.save('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/label.npy', labels)
np.save('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/start.npy', start)
np.save('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/end.npy', end)

'''

#3/3/2025 replaced
import pandas as pd
import os
import numpy as np
import nibabel as nib

labels_df = pd.read_csv('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/name_mapping_pre-processing.csv')
df = labels_df[['Grade', 'BraTS_2020_subject_ID']]

rootdir = 'D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/'
subdirs = os.listdir(rootdir)
label = np.zeros(16)
inputs = np.zeros((16, 160, 210, 100, 1))
label_2d = np.zeros(1600)
inputs_2d = np.zeros((1600, 160, 210, 1))
i = 0
j = 0
for sub in subdirs:
    file_path = rootdir + '/' + sub + '/' + sub + '_weighted1233.nii.gz'
    if os.path.exists(file_path):
        nii_img = nib.load(file_path)
        wt = nii_img.get_fdata()
        intermd = wt[40:200, 20:230, 27:127]
        inputs[i, :, :, :, 0] = intermd

        grade = df.loc[df['BraTS_2020_subject_ID'] == sub, 'Grade'].values[0]  
        if grade == 'HGG':
            label[i] = 1
        else:
            label[i] = 0

        for t in range(100):
            inputs_2d[j, :, :, 0] = intermd[:, :, t]
            label_2d[j] = label[i]
            j += 1
        i += 1
#(369, 160, 210, 100, 1)
#369
print(inputs.shape)
print(label.shape)

print(inputs_2d.shape)
print(label_2d.shape)



np.save('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/inputs.npy', inputs)
np.save('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/label.npy', label)

np.save('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/inputs_2d.npy', inputs_2d)
np.save('D:/CSUSM-Computer Science/DL/Deep Learning/starter_code/starter_code/Brain_Tumor_Data/label_2d.npy', label_2d)
'''