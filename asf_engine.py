import cv2
from asf.asf_struct import *
from asf.asf_common import *
import os
import ctypes
import datetime
import binascii
import logging
import numpy as np
from ctypes import memmove

logging.basicConfig(
    filename='face_engine.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FaceEngine:
    def __init__(self):
        self.app_id = b"Gfjscyy7VavLrQH99ayXg8LPmNcNfyuF9CooFmEbWPYr"
        self.sdk_key = b"8KkzFD9FZ9JTDSeL9HoJ4H7KJiViFZZDrgFiuxs5zSev"
        self.active_key = b"82G1-11V1-6124-6N2P"
        self.active_data = "/home/ubuntu/AIClass/ArcFacePro64.dat"
        self.video_engine = self.initEngine()
    def initEngine(self):
        try:
            if os.path.exists(self.active_data):
                logging.info("使用离线激活")
                ret = offline_activate(c_char_p(self.active_data.encode('utf-8')))
            else:
                logging.info("使用在线激活")
                ret = online_activate(self.app_id, self.sdk_key, self.active_key)
            
            if ret != 0 and ret != 90114:
                logging.error(f"激活失败: {ret}")
                return None

            video_engine = c_void_p()
            video_mask = ASF_FACE_DETECT | ASF_FACERECOGNITION
            video_ret = init_engine(ASF_DETECT_MODE_IMAGE, 0x5, 50, video_mask, byref(video_engine))
            
            if video_ret != 0:
                logging.error(f"视频模式引擎初始化失败: {video_ret}")
                return None
            
            logging.info(f"引擎初始化成功: {video_ret}")
            return video_engine
        except Exception as e:
            logging.error(f"初始化引擎时发生错误: {str(e)}")
            return None

    def align_image_width(self, img):
        if img is None:
            logging.error("图片加载失败")
            return None

        width, height = img.shape[1], img.shape[0]
        remainder = width % 4
        if remainder != 0:
            crop_width = width - remainder
            img = img[:, :crop_width]
            logging.info(f"图像宽度剪裁 {remainder} 个像素，新的宽度为 {crop_width}")
        else:
            logging.info(f"图像宽度已经是4的倍数，无需剪裁 {width} {height}")

        return img

    def compare_faces(self, feature1, feature2):
        if feature1 is None or feature2 is None:
            logging.error("特征为空")
            return 0
            
        try:
            if feature1.featureSize <= 0 or feature2.featureSize <= 0:
                logging.error("特征大小无效")
                return 0
                
            similarity = c_float()
            compare_ret = faceFeatureCompare(self.video_engine, feature1, feature2, byref(similarity), 0x2)
                                           
            if compare_ret == 0:
                return similarity.value
            else:
                logging.error(f"比对失败，错误码：{compare_ret}")
                return 0
        except Exception as e:
            logging.error(f"比对过程中发生错误: {str(e)}")
            return 0
    def detect_and_get_face_featureOne(self, img):
        try:
            if img is None:
                logging.error("图片加载失败")
                return None

            img = self.align_image_width(img)
            if img is None:
                return None

            # 确保图像数据是连续的
            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)

            image_bytes = bytes(img)
            image_buffer = (c_ubyte * len(image_bytes))()
            memmove(image_buffer, image_bytes, len(image_bytes))
            image_ubytes = cast(image_buffer, POINTER(c_ubyte))

            detect_faces = ASF_MultiFaceInfo()
            ret = detect_face(self.video_engine, img.shape[1], img.shape[0],
                            ASVL_PAF_RGB24_B8G8R8, image_ubytes, byref(detect_faces), 0x1)

            if ret != 0:
                logging.error(f"检测人脸失败：{ret}")
                return None

            logging.info(f"返回人脸: {ret}")

            if detect_faces.faceNum <= 0:
                logging.error("未检测到人脸")
                return None

            # 只处理第一个人脸
            single = ASF_SingleFaceInfo()
            single.faceRect = detect_faces.faceRect[0]
            single.faceOrient = detect_faces.faceOrient[0]
            single.faceDataInfo = detect_faces.faceDataInfoList[0]

            feature = ASF_FaceFeature()
            f2ret = faceFeatureExtract(self.video_engine,
                                     img.shape[1],
                                     img.shape[0],
                                     ASVL_PAF_RGB24_B8G8R8,
                                     image_ubytes,
                                     single,
                                     0x0,
                                     0x0,
                                     byref(feature))

            logging.info(f"返回特征：{f2ret}")

            if f2ret != 0:
                logging.error(f"特征提取失败：{f2ret}")
                return None

            # 复制特征数据到新的内存位置
            return self.copy_feature(feature)

        except Exception as e:
            logging.error(f"单人脸特征提取过程中发生错误: {str(e)}")
            return None
        
    def detect_and_get_face_features(self, img):
        image_buffer = None
        try:
            path = f'/data/pictures/{datetime.datetime.now().strftime("%Y%m%d")}'
            if not os.path.exists(path):
                os.makedirs(path)

            if img is None:
                logging.error("输入图片为空")
                return []

            features = []
            img = self.align_image_width(img)
            if img is None:
                return []

            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)

            # 使用 create_string_buffer 替代原来的数组分配
            image_buffer = ctypes.create_string_buffer(bytes(img))
            image_ubytes = ctypes.cast(image_buffer, POINTER(c_ubyte))

            detect_faces = ASF_MultiFaceInfo()
            ret = detect_face(self.video_engine, img.shape[1], img.shape[0],
                            ASVL_PAF_RGB24_B8G8R8, image_ubytes, byref(detect_faces), 0x1)

            if ret != 0:
                logging.error(f"人脸检测失败: {ret}")
                return []

            if detect_faces.faceNum > 0:
                face_rects = cast(detect_faces.faceRect,
                                POINTER(MRECT * detect_faces.faceNum)).contents
                face_orients = cast(detect_faces.faceOrient,
                                  POINTER(c_int * detect_faces.faceNum)).contents
                face_data_infos = cast(detect_faces.faceDataInfoList,
                                     POINTER(ASF_FaceDataInfo * detect_faces.faceNum)).contents

                for i in range(detect_faces.faceNum):
                    try:
                        single = ASF_SingleFaceInfo()
                        single.faceRect = face_rects[i]
                        single.faceOrient = face_orients[i]
                        single.faceDataInfo = face_data_infos[i]

                        feature = ASF_FaceFeature()
                        f2ret = faceFeatureExtract(self.video_engine,
                                                 img.shape[1],
                                                 img.shape[0],
                                                 ASVL_PAF_RGB24_B8G8R8,
                                                 image_ubytes,
                                                 single,
                                                 0x0,
                                                 0x0,
                                                 byref(feature))

                        if f2ret == 0:
                            feature_copy = self.copy_feature(feature)
                            if feature_copy:
                                features.append(feature_copy)
                            
                            cv2.rectangle(img,
                                        (face_rects[i].left, face_rects[i].top),
                                        (face_rects[i].right, face_rects[i].bottom),
                                        (0, 255, 0), 2)
                        else:
                            logging.error(f"特征提取失败，错误码: {f2ret}")

                    except Exception as e:
                        logging.error(f"处理单个人脸时发生错误: {str(e)}")
                        continue

                try:
                    filename = f'{path}/arc_detect-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'
                    cv2.imwrite(filename, img)
                    logging.info(f"保存检测结果图片到: {filename}")
                except Exception as e:
                    logging.error(f"保存图片失败: {str(e)}")

            return features

        except Exception as e:
            logging.error(f"人脸特征提取过程中发生错误: {str(e)}")
            return []
        finally:
            # 确保清理分配的内存
            image_buffer = None

    def copy_feature(self, feature):
        try:
            if feature.featureSize <= 0:
                return None

            new_feature = ASF_FaceFeature()
            new_feature.featureSize = feature.featureSize
            
            # 使用 create_string_buffer 来确保内存被正确管理
            new_buffer = ctypes.create_string_buffer(feature.featureSize)
            ctypes.memmove(new_buffer, feature.feature, feature.featureSize)
            new_feature.feature = ctypes.cast(new_buffer, POINTER(c_ubyte))
            
            return new_feature
        except Exception as e:
            logging.error(f"复制特征数据时发生错误: {str(e)}")
            return None

    def store_blob_to_feature(self, blob_data):
        if not blob_data:
            logging.error("Blob数据为空")
            return None

        try:
            feature = ASF_FaceFeature()
            feature.featureSize = len(blob_data)
            
            # 创建新的缓冲区并安全地复制数据
            buffer = (c_ubyte * len(blob_data))()
            memmove(buffer, blob_data, len(blob_data))
            feature.feature = buffer
            
            return feature
        except Exception as e:
            logging.error(f"创建特征时发生错误: {str(e)}")
            return None

    def printFeature(self, feature):
        try:
            if feature is None:
                logging.error("特征为空，无法打印")
                return
                
            print("Feature Size:", feature.featureSize)
            feature_data = feature.feature[:feature.featureSize]
            binary_string = ''.join(format(byte, '02x') for byte in feature_data)
            print("Feature Data (Binary String):", binary_string)
        except Exception as e:
            logging.error(f"打印特征时发生错误: {str(e)}")

    def read_binary_file(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                return file.read()
        except Exception as e:
            logging.error(f"读取文件失败: {str(e)}")
            return None

    def create_asf_face_data_info_from_bin(self, file_path):
        binary_data = self.read_binary_file(file_path)
        if binary_data:
            return self.create_asf_face_data_info_from_blob(binary_data)
        return None
    
    def create_asf_face_data_info_from_blob(self, binary_data):
        try:
            if not binary_data:
                logging.error("二进制数据为空")
                return None
                
            buffer = (ctypes.c_ubyte * len(binary_data))()
            memmove(buffer, binary_data, len(binary_data))
            
            face_data_info = ASF_FaceDataInfo()
            face_data_info.data = ctypes.addressof(buffer)
            face_data_info.dataSize = len(binary_data)
            
            return face_data_info
        except Exception as e:
            logging.error(f"创建人脸数据信息时发生错误: {str(e)}")
            return None
    def __del__(self):
        try:
            if self.video_engine:
                uninit_engine(self.video_engine)
                logging.info("引擎资源已释放")
        except Exception as e:
            logging.error(f"释放引擎资源时发生错误: {str(e)}")
# 使用示例
if __name__ == "__main__":
    try:
        face_engine = FaceEngine()
        # 这里可以添加测试代码
        print("FaceEngine初始化成功")
    except Exception as e:
        logging.error(f"主程序运行错误: {str(e)}")