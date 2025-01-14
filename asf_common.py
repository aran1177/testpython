from ctypes import *
from enum import Enum
import platform
from ctypes import CDLL
from asf.asf_struct import *


# 判断当前平台
if platform.system() == 'Linux':
    face_dll = CDLL("libarcsoft_face.so")
    face_engine_dll = CDLL("libarcsoft_face_engine.so")
else:
    face_dll = CDLL("libarcsoft_face.dll")
    face_engine_dll = CDLL("libarcsoft_face_engine.dll")


#====================常量类型定义====================
ASF_DETECT_MODE_VIDEO    = 0x00000000        #   视频流检测模式
ASF_DETECT_MODE_IMAGE    = 0xFFFFFFFF        #   图片检测模式

ASF_NONE				 = 0x00000000        #   无属性
ASF_FACE_DETECT			 = 0x00000001        #   此处detect可以是tracking或者detection两个引擎之一，具体的选择由detect mode 确定
ASF_FACERECOGNITION		 = 0x00000004        #   人脸特征
ASF_AGE					 = 0x00000008        #   年龄
ASF_GENDER				 = 0x00000010        #   性别
ASF_FACE3DANGLE			 = 0x00000020        #   3D角度
ASF_FACELANDMARK		 = 0x00000040        #   额头区域检测
ASF_LIVENESS			 = 0x00000080        #   RGB活体
ASF_IMAGEQUALITY		 = 0x00000200        #   图像质量检测
ASF_IR_LIVENESS			 = 0x00000400        #   IR活体
ASF_FACESHELTER			 = 0x00000800        #   人脸遮挡
ASF_MASKDETECT			 = 0x00001000        #   口罩检测
ASF_UPDATE_FACEDATA		 = 0x00002000        #   人脸信息

ASVL_PAF_RGB24_B8G8R8    = 0x201             #   图片格式


#检测时人脸角度的优先级--枚举类型
class ArcSoftFaceOrientPriority(Enum):
    ASF_OP_0_ONLY = 0x1,                      #   常规预览下正方向
    ASF_OP_90_ONLY = 0x2,                     #   基于0°逆时针旋转90°的方向
    ASF_OP_270_ONLY = 0x3,                    #   基于0°逆时针旋转270°的方向
    ASF_OP_180_ONLY = 0x4,                    #   基于0°旋转180°的方向（逆时针、顺时针效果一样）
    ASF_OP_0_HIGHER_EXT = 0x5,                #   全角度

#==================================================
class ASF_CompareModel(Enum):
	ASF_LIFE_PHOTO = 0x1	# 用于生活照之间的特征比对，推荐阈值0.80
	ASF_ID_PHOTO = 0x2		# 用于证件照或生活照与证件照之间的特征比对，推荐阈值0.82
	

#====================Api接口映射定义====================
#主要定义函数的入参与返回类型



#在线激活Api
online_activate = face_engine_dll.ASFOnlineActivation
online_activate.restype = c_int32
online_activate.argtypes = (c_char_p, c_char_p, c_char_p)
#离线激活Api
offline_activate = face_engine_dll.ASFOfflineActivation
offline_activate.restype = c_int32
offline_activate.argtypes = [c_char_p]

#引擎初始化Api
init_engine = face_engine_dll.ASFInitEngine
init_engine.restype = c_int32
init_engine.argtypes = (c_long, c_int32, c_int32, c_int32, POINTER(c_void_p))

#人脸检测Api
detect_face = face_engine_dll.ASFDetectFaces
detect_face.restype = c_int32
detect_face.argtypes = (c_void_p, c_int32, c_int32, c_int32, POINTER(c_ubyte), POINTER(ASF_MultiFaceInfo),c_int32)
#人脸对比
faceFeatureCompare = face_engine_dll.ASFFaceFeatureCompare
faceFeatureCompare.restype = c_int32
faceFeatureCompare.argtypes = (c_void_p, POINTER(ASF_FaceFeature), POINTER(ASF_FaceFeature), POINTER(c_float),c_int32)

uninit_engine = face_engine_dll.ASFUninitEngine
uninit_engine.restype = c_int32
uninit_engine.argtypes = (c_void_p)
#特征提取
# [in] 引擎handle
# [in] 图片宽度
# [in] 图片高度
# [in] 颜色空间格式
# [in] 图片数据
# [in] 单张人脸位置和角度信息
# [in] 注册 1 识别为 0
# [in] 带口罩 1，否则0
# [out] 人脸特征
faceFeatureExtract = face_engine_dll.ASFFaceFeatureExtract
faceFeatureExtract.restype = c_int32
faceFeatureExtract.argtypes = (c_void_p,c_int32 ,c_int32,c_int32,POINTER(c_ubyte), POINTER(ASF_SingleFaceInfo), c_int32,c_int32,POINTER(ASF_FaceFeature))

