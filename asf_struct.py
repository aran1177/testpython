from ctypes import *

# 结构体
class MRECT(Structure):
	"""人脸框"""
	_fields_ = [
        ("left", c_int),
        ("top", c_int),         
        ("right", c_int),
        ("bottom", c_int)
    ]


class ASF_Face3DAngleInfo(Structure):
	"""3D角度信息"""
	_fields_ = [
		("roll", POINTER(c_float)),
		("yaw", POINTER(c_float)),
		("pitch", POINTER(c_float))
    ]
class ASF_FaceDataInfo(Structure):
	"""人脸信息"""
	_fields_ = [
		("data", c_void_p), # 人脸信息
		('dataSize', c_int) # 人脸信息长度
    ]
	
class ASF_MultiFaceInfo(Structure):
	"""多人脸框信息"""
	_fields_ = [
        ("faceNum", c_int),                              # 检测到的人脸个数
        ("faceRect", POINTER(MRECT)),                    # 人脸框信息
        ("faceOrient", POINTER(c_int)),                  # 人脸图像的角度，可以参考 ASF_OrientCode
        ("faceID", POINTER(c_int)),                      # face ID
        ("faceDataInfoList", POINTER(ASF_FaceDataInfo)), # 人脸检测信息
        ("faceIsWithinBoundary", POINTER(c_int)),        # 人脸是否在边界内 0 人脸溢出；1 人脸在图像边界内
        ("foreheadRect", POINTER(MRECT)),                # 人脸额头区域
        ("face3DAngleInfo", ASF_Face3DAngleInfo)         # 人脸3D角度
    ]

class ASF_SingleFaceInfo(Structure):
	"""单人脸信息"""
	_fields_ = [
		("faceRect", MRECT),                # 人脸框信息
		("faceOrient", c_int),              # 人脸图像角度，可以参考 ASF_OrientCode
		("faceDataInfo", ASF_FaceDataInfo)  # 单张人脸信息
    ]

class ASF_FaceFeature(Structure):
	"""人脸特征信息"""
	_fields_ = [
		("feature", POINTER(c_ubyte)),        # 人脸特征信息
		("featureSize", c_int)                # 人脸特征信息长度 
    ]
