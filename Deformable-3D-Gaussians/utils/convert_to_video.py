import cv2
import os
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)
def create_video_from_images(folder_path, output_file, frame_rate=30, img_size=None): #default = 30
    images = [img for img in os.listdir(folder_path) if img.endswith(".jpg") or img.endswith(".png")]
    images = sorted_alphanumeric(images)
    frame = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, layers = frame.shape

    if img_size is not None:
        width, height = img_size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

    for image in images:
        img = cv2.imread(os.path.join(folder_path, image))
        if img_size is not None:
            img = cv2.resize(img, img_size)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

folder_path = '/home/ad/20813716/Deformable-3D-Gaussians/output/exp_nerfds_plate-47/test/ours_40000/renders'  
output_file = '/home/ad/20813716/Deformable-3D-Gaussians/output/exp_nerfds_plate-47/test/ours_40000/output_videos.mp4'  
create_video_from_images(folder_path, output_file)
