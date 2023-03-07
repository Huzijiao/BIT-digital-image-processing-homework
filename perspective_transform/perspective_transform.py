import tkinter as tk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter.filedialog

# 实现openCV中的库函数getPerspectiveTransform，源图像中待测矩形的四点坐标，目标图像中矩形的四点坐标，返回由源图像中矩形到目标图像矩形变换的矩阵
def getPerspectiveTransform(sourcePoints, destinationPoints):
    if sourcePoints.shape != (4,2) or destinationPoints.shape != (4,2):
        raise ValueError("There must be four source points and four destination points")
    a = np.zeros((8, 8))
    b = np.zeros((8))
    for i in range(4):
        a[i][0] = a[i+4][3] = sourcePoints[i][0]
        a[i][1] = a[i+4][4] = sourcePoints[i][1]
        a[i][2] = a[i+4][5] = 1
        a[i][3] = a[i][4] = a[i][5] = 0
        a[i+4][0] = a[i+4][1] = a[i+4][2] = 0
        a[i][6] = -sourcePoints[i][0]*destinationPoints[i][0]
        a[i][7] = -sourcePoints[i][1]*destinationPoints[i][0]
        a[i+4][6] = -sourcePoints[i][0]*destinationPoints[i][1]
        a[i+4][7] = -sourcePoints[i][1]*destinationPoints[i][1]
        b[i] = destinationPoints[i][0]
        b[i+4] = destinationPoints[i][1]
    x = np.linalg.solve(a, b)
    x.resize((9,), refcheck=False)
    x[8] = 1 
    return x.reshape((3,3))



# 将list存储的坐标点对存入4*2的数组中
def order_points(pts):
	x = np.zeros((4,2), dtype="float32")
	for i in range(len(pts)):
		x[i] = pts[i] 
	return x

# 透视变换
def perspective_transformation(img, pts):
	rect = order_points(pts)
	# 赋值给 top-left top-right bottom-right bottom-left
	(tl, tr, bl, br) = rect
	print(rect)
	# 计算新图像的宽
	widthT = np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2)
	widthB = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)
	# 计算新图像的高
	heightL = np.sqrt((tl[0]-bl[0])**2 + (tl[1]-bl[1])**2)
	heightR = np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2)
	# 找到最大的高和宽
	maxWidth = max(int(widthB), int(widthT))
	maxHeight = max(int(heightL), int(heightR))
	print('width:'+str(maxWidth))
	print('height:'+str(maxHeight))
	dst = np.float32([[0,0],[maxWidth-1,0],[maxWidth-1, maxHeight-1],[0,maxHeight-1]])
	# 计算变换矩阵
	M = getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
	return warped

def draw(event, x, y, flags, param):
	global btn_down
	# 鼠标点击事件允许产生4次，每点击一次画一个圆点
	if event == cv2.EVENT_LBUTTONDBLCLK and len(pts)<4: 
		btn_down = True
		# 画圆(img, center, radius, color[, thickness[, lineType[, shift]]])
		cv2.circle(image,(x,y), 5, (255,0,0), -1)
		if len(pts)>0:
			cv2.line(image, pts[-1], (x, y), (0,0,0), 2)
		cv2.imshow('Draw',image)
		param=(x,y)
		pts.append(param)
	elif len(pts)>=4:
		cv2.line(image, pts[-1], pts[0], (0,0,0), 2)
		cv2.putText(image,'Press any key to exit', (5,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1,cv2.LINE_AA )
		cv2.imshow('Draw',image)
	return

# 创建一个窗口
root = tk.Tk()
# 读取图像
img = Image.open('gray.jpg')
image = np.array(img)
# image = cv2.imread('gray.jpg')
# image = np.array(image)
cv2.namedWindow('Draw')
cv2.imshow('Draw',image)
# 用list存储点击捕获的点坐标
pts = list()
pts.append(cv2.setMouseCallback('Draw', draw))
del pts[0]
print(pts)
cv2.waitKey(0)
cv2.line(image, pts[-1], pts[0], (0,0,0), 1)
cv2.imshow('Draw',image)
cv2.destroyAllWindows()
print (pts)
pts = np.float32(pts)
warped = perspective_transformation(image,pts)
h,w = warped.shape
# 创建画布
canvas = tk.Canvas(root, width=w, height=h)
canvas.pack()
# 把Numpy ndarray转变为PhotoImage，在画布上添加图像
photo = ImageTk.PhotoImage(image=Image.fromarray(warped))
canvas.create_image(0,0,image=photo, anchor=tk.NW)
root.mainloop()
# 展示结果
plt.subplot(121),plt.imshow(image,cmap='Greys_r'),plt.title('Input Image')
plt.subplot(122),plt.imshow(warped,cmap='Greys_r'),plt.title('Output Image')
plt.show()
