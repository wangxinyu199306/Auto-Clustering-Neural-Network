import cv2
import numpy as np

def visual_result(image_list,pos_list):
    image_list.astype(np.uint8)
    paint_background = np.zeros((2048,2048,3), np.uint8) + 255
    print (pos_list)
    max_pos = pos_list.max(axis = 0)
    hori_coe = 2000.0/max_pos[0][0]
    verti_coe = 2000.0/max_pos[0][1]

    for i in range(image_list.shape[0]):
        temp_pos_x = int(hori_coe * pos_list[i,0,0])
        temp_pos_y = int(verti_coe * pos_list[i,0,1])
        if(temp_pos_x+32<2048) and (temp_pos_y+32<2048):
            paint_background[temp_pos_x:temp_pos_x+32,temp_pos_y:temp_pos_y+32,:] = image_list[i,:,:,:]
    
    cv2.imwrite("predict.png",paint_background)



if __name__ == '__main__':
    pos_list = np.zeros((10,1,2), np.float32)
    pos_list_rand = np.random.rand(10,1,2)
    visual_result(pos_list,pos_list_rand)