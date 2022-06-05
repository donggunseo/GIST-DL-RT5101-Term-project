import cv2,os
from tqdm import tqdm

def crop_train(og_dir_path, cropped_dir_path):
    smoking_file = os.listdir(os.path.join(og_dir_path, 'smoking'))
    nonsmoking_file = os.listdir(os.path.join(og_dir_path, 'nonsmoking'))
    os.makedirs(os.path.join(cropped_dir_path, 'smoking'), exist_ok=True)
    os.makedirs(os.path.join(cropped_dir_path, 'nonsmoking'), exist_ok=True)
    for file in tqdm(smoking_file):
        img = cv2.imread(os.path.join(og_dir_path,'smoking',file), cv2.IMREAD_UNCHANGED)
        cropped_img = img[998:, 998-100:(998*2)+100]
        cv2.imwrite(os.path.join(cropped_dir_path,'smoking',file), cropped_img)
    for file in tqdm(nonsmoking_file):
        img = cv2.imread(os.path.join(og_dir_path,'nonsmoking',file), cv2.IMREAD_UNCHANGED)
        cropped_img = img[998:, 998-100:(998*2)+100]
        cv2.imwrite(os.path.join(cropped_dir_path,'nonsmoking',file), cropped_img)

if __name__ == '__main__':
    og_dir_path = '../data/Train_image_original'
    cropped_dir_path = '../data/Train_image_cropped'
    crop_train(og_dir_path, cropped_dir_path)