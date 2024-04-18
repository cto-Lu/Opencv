'''
Author: CloudSir
@Github: https://github.com/CloudSir
Date: 2022-05-13 10:24:00
LastEditTime: 2022-07-27 11:47:08
LastEditors: CloudSir
Description: 
'''
import cv2

def get_DroidCam_url(ip, port=4747, res='480p'):
    res_dict = {
        '240p': '320x240',
        '480p': '640x480',
        '720p': '1280x720',
        '1080p': '1920x1080',  
    }
    url = f'http://{ip}:{port}/mjpegfeed?{res_dict[res]}'
    return url


# DroidCam 显示的IP地址、端口号和相机分辨率（可选 240p,480p,720p,1080p）
cap = cv2.VideoCapture(get_DroidCam_url('192.168.43.212', 4747, '720p'))

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('image', frame)

    key = cv2.waitKey(1)
    # 按q退出程序
    if key == ord('q'):
        break

# 释放VideoCapture
cap.release()
# 销毁所有的窗口
cv2.destroyAllWindows()

