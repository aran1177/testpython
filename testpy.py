import os
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Point
import cv2
import torch
import numpy as np
from sensor_msgs.msg import  CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO
from asf.asf_engine import FaceEngine
#from asf.asf_engine import FaceEngine
from aiclass_msgs.msg import PersonMessage,Persons
from message_filters import ApproximateTimeSynchronizer, Subscriber
from .databaseconn import get_mysqldb_connection,getIp
import datetime
from std_msgs.msg import String
import logging
import subprocess
import signal
import sys
import psutil
import atexit
import binascii


class StudenObject:
    sno = ''
    feature = ''
    type = 100
    id = 1
    online=0
    lasttime = 0#最后一次出现的时间
    def __init__(self,id, sno, feature):
        self.sno = sno
        self.id = id
        self.feature = feature
        self.type = 100
        self.online=0
    def __str__(self):
        return f"学号: {self.sno}, 特征: {self.feature}, 类型: {self.type} id: {self.id}"

class ObjectDetection(Node):
    def __init__(self):
        super().__init__("ObjectDetection")

        self.declare_parameter("classid", 1, ParameterDescriptor(description="class table id"))
        self.classid = self.get_parameter("classid").get_parameter_value().integer_value

        self.declare_parameter("serverid", 0, ParameterDescriptor(description="serverid"))
        self.serverid = self.get_parameter("serverid").get_parameter_value().integer_value

        self.conn = get_mysqldb_connection()
        print('连接sql')
        self.insertTime=15
        
        self.yolo_teacher=  YOLO('/home/ubuntu/AIClass/teacher.pt')
        self.yolo_stu=      YOLO('/home/ubuntu/AIClass/stu-8.pt')
        self.yolo_v8=       YOLO('/home/ubuntu/AIClass/yolo11m.pt')
        #self.yolo_face=     YOLO('/home/ubuntu/AIClass/yoloface.pt')
        print('init yolo model success')
        # 初始化图片模式引擎
        self.face_engine = FaceEngine()
        print('init face engine success')
        # camerainfo的id就可以拼接相机的节点名字，f'/camera_{id}/compressed'然后创建一个subscriber数组，
        self.cameras = self.get_camera_list_by_classid(self.classid)
        self.subscribers = []
        self.indeximg=0
        # Publishers
        self.pub_person = self.create_publisher(Persons, f'/person_{self.classid}', 10)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.half=False
        # Realsense package
        self.bridge = CvBridge()

        # Subscribers
        self.create_subscriptions()
        #self.createZK()
        self.students = []#数据不在从一开始就添加了，要改成获取
        self.initStudents()
        self.create_subscriptions_kaoqin()
        
        self.initTeachers()

        self.timer=1
        self.update_timer=self.create_timer(60, self.update_nodes)  # 每1分钟执行一次

        self.pid = os.getpid()
        self.parent_pid = os.getppid()

        self.path=f'/data/pictures/{datetime.datetime.now().strftime("%Y%m%d")}'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        # 注册信号处理器
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # 注册退出处理器
        atexit.register(self._cleanup)
        self.class_counts = {i: 0 for i in range(6)}  # 假设class_id的范围是0到5

        self.class_img_path=''
        self.imgs=[]
    def hex_string_to_binary_data(self,hex_string):
        try:
            binary_data = binascii.unhexlify(hex_string)
            return binary_data
        except binascii.Error as err:
            print("十六进制字符串转换错误:", err)
            return None

    def update_nodes(self):
        
        try:
            for img in self.imgs:
                self.taskImage(img)
            # 打印每个类别的计数
            cursor=self.conn.cursor()
            for class_id, count in self.class_counts.items():
                self.get_logger().info(f"Class {class_id} count: {count}")
                # 插入数据库
                
                cursor.execute("INSERT INTO ai_behavior (bv,num,total, time, classid_id, timer,img) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                            ( class_id,count,len(self.students),  (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),self.classid,\
                            self.timer,self.class_img_path))
                self.conn.commit()  # 提交事务
            self.conn.commit()  # 提交事务

            #每隔15分钟提交一次考勤数据
            
                # 插入数据库
            for student in self.students:
                cursor = self.conn.cursor()
                
                if student.online==0 and student.lasttime-self.timer>self.insertTime:
                    type_value = 0
                else:
                    type_value = student.online
                if student.lasttime%self.insertTime > 0:
                    type_value = student.online
                self.get_logger().info(f"type_value: {type_value}")

                imagepath=self.class_img_path.replace('/data/', getIp(self.serverid))
                cursor.execute("INSERT INTO ai_kaoqin (sno,  timer, classid_id, time, creator_id,img,online,description) VALUES \
                                (%s,  %s, %s, %s, %s, %s,%s, %s)",
                        (student.sno,  self.timer, self.classid, \
                         (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), student.id,imagepath,type_value,(1 if self.timer%self.insertTime==0 else 0))
                    )
                
            self.conn.commit()  # 提交事务
            cursor.close()

            
            
        except Exception as e:
            self.get_logger().error(f"Error process image: {str(e)}")
        finally:
            self.timer+=1
            self.class_counts = {i: 0 for i in range(6)} 
            self.imgs=[]
            cursor.close()

        self.updateStudents()


        if self.conn.is_connected():
            cursor = self.conn.cursor()
            cursor.execute("SELECT status FROM classtable WHERE id=%s",(self.classid,))
            tmp1=cursor.fetchone()
            if tmp1[0]>=2:
                self.destroy_node()
    def destroy_node(self):
        self.get_logger().info(f'时间到,object：课id{self.classid}-')
        if self.conn is not None:
            self.conn.close()
        del self.yolo_teacher
        del self.yolo_stu
        del self.face_engine
        del self.yolo_v8
        self.update_timer.cancel()

        self.get_logger().info('Reached max count, shutting down...')
        self._cleanup()
        super().destroy_node()
        sys.exit(0)
    def _signal_handler(self, signum, frame):
        """处理终止信号"""
        self.get_logger().info(f'Received signal {signum}')
        self._cleanup()
        sys.exit(0)
    def create_subscriptions_kaoqin(self):
        self.get_logger().info(f'create_subscriptions_kaoqin{self.classid}') 
        self.subscription = self.create_subscription(
            String,
            f'/kaoqin_{self.classid}',
            self.listener_callback,
            10)
        self.subscription  

    def listener_callback(self, msg):
        try:
            self.get_logger().info('from kaoqin: %s' % msg.data)
            if msg.data in self.students.sno:
                return
            else:
                cursor = self.conn.cursor()
                cursor.execute("select u.id,u.username,f.Feature1 from system_userinfo u left join faceid f on f.StudentNum=u.username where u.username=%s", (msg.data,))
                row = cursor.fetchone()
                if row:
                    student = StudenObject(row[0], row[1], row[2])
                    self.students.append(student)
                    self.get_logger().info(f'add student:{msg.data}')
                cursor.close()
        except Exception as e:
            self.get_logger().error(f'Error in listener_callback: {e}')
        finally:
            self.conn.close()
            
    def initTeachers(self):
        self.teachers = []
        cursor = self.conn.cursor()
        cursor.execute("select u.id,u.username,f.Feature1 from system_userinfo u left join faceid f on u.username=f.studentnum  where u.id in ( SELECT id FROM classtable_teachers where classtable_id= %s)", (self.classid,))
        rows = cursor.fetchall()
        for row in rows:
            teacher = StudenObject(row[0], row[1], row[2])
            self.teachers.append(teacher)
        cursor.close()
        self.get_logger().info(f"teacher:{self.teachers[0].sno} inited")

    def initStudents(self):
        
        cursor = self.conn.cursor()
        cursor.execute("select u.id,u.username,f.Feature1 from system_userinfo u left join faceid f on u.username=f.StudentNum  where u.id in ( SELECT id FROM classtable_students where classtable_id= %s)", (self.classid,))
        rows = cursor.fetchall()
        for row in rows:
            if row[2] is None:
                continue
            hex_str = row[2].decode('ascii')
            binary_data = bytes.fromhex(hex_str)
            featur=self.face_engine.store_blob_to_feature(binary_data)
            #self.face_engine.printFeature(featur)
            if featur is None:
                continue
            student = StudenObject(row[0],row[1], featur)
            self.students.append(student)
        cursor.close()
        self.get_logger().info(f'init student:{len(self.students)}')

    def updateStudents(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT u.id, u.username, f.Feature1 FROM system_userinfo u "
                "LEFT JOIN faceid f ON u.username=f.StudentNum "
                "WHERE u.id IN (SELECT id FROM classtable_students WHERE classtable_id=%s)",
                (self.classid,)
            )
            rows = cursor.fetchall()
            
            # 创建现有学生字典，用于快速查找
            existing_students = {student.id: student for student in self.students}
            db_student_ids = set()
            
            # 处理每一行数据
            for row in rows:
                student_id = row[0]
                username = row[1]
                feature_blob = row[2]
                db_student_ids.add(student_id)
                
                try:
                    feature = self.face_engine.store_blob_to_feature(feature_blob)
                    
                    if student_id not in existing_students:
                        # 添加新学生
                        new_student = StudenObject(student_id, username, feature)
                        self.students.append(new_student)
                        self.get_logger().info(f'添加新学生: {username} (ID: {student_id})')
                        
                except Exception as e:
                    self.get_logger().error(f'Error processing student {username}: {str(e)}')
            
            # 移除不存在的学生
            self.students = [student for student in self.students if student.id in db_student_ids]
            
            self.get_logger().info(f'学生更新完成，总数: {len(self.students)}')
            
        except Exception as e:
            self.get_logger().error(f'学生更新出错: {str(e)}')
        finally:
            if cursor:
                cursor.close()


    def splitDect(self, img):
        """
        对图像进行切片检测并合并重叠框，确保完整覆盖图像
        """
        all_results = []
        height, width, channels = img.shape  # 注意：shape的顺序是(h,w,c)

        # 设置切片参数
        slice_height = 640
        slice_width = 640
        overlap = 100
        rgb_image = img.copy()  # 创建副本以避免修改原图
        raw_detections = []

        # 计算需要的切片数量
        num_slices_h = max(1, (height + slice_height - overlap - 1) // (slice_height - overlap))
        num_slices_w = max(1, (width + slice_width - overlap - 1) // (slice_width - overlap))

        for i in range(num_slices_h):
            for j in range(num_slices_w):
                # 计算当前切片的坐标
                start_y = min(i * (slice_height - overlap), max(0, height - slice_height))
                start_x = min(j * (slice_width - overlap), max(0, width - slice_width))
                
                # 确保最后一个切片能够覆盖到边界
                end_y = min(start_y + slice_height, height)
                end_x = min(start_x + slice_width, width)
                
                # 提取切片
                slice_img = img[start_y:end_y, start_x:end_x]
                
                # 检查切片有效性
                if slice_img.shape[0] < 20 or slice_img.shape[1] < 20:  # 设置最小有效尺寸
                    continue

                # YOLOv8 检测
                results = self.yolo_v8(slice_img, conf=0.6, iou=0.6, verbose=False)

                # 处理检测结果
                for result in results[0].boxes.data.clone():
                    x1, y1, x2, y2, conf, class_id = result
                    if class_id == 0:  # 只处理人类目标
                        # 转换到原图坐标系
                        x1, y1, x2, y2 = (
                            int(x1 + start_x),
                            int(y1 + start_y),
                            int(x2 + start_x),
                            int(y2 + start_y)
                        )
                        
                        # 确保坐标不超出图像边界
                        x1 = max(0, min(x1, width))
                        y1 = max(0, min(y1, height))
                        x2 = max(0, min(x2, width))
                        y2 = max(0, min(y2, height))
                        
                        if (x2 - x1) * (y2 - y1) > 0:  # 确保框面积大于0
                            raw_detections.append((x1, y1, x2, y2, float(conf), int(class_id)))
                            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # 在每一行结束时更新进度
        #self.get_logger().info(f'处理进度: {(i+1)/num_slices_h*100:.1f}%')

        # 保存检测结果图像
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        output_path = os.path.join(self.path, f'result2_{self.classid}_{self.timer}_{timestamp}.jpg')
        cv2.imwrite(output_path, rgb_image)
        
        # 合并重叠框
        filtered_results = self._merge_overlapping_boxes(raw_detections, img)
        
        self.get_logger().info(f'原始检测数量: {len(raw_detections)}, 过滤后数量: {len(filtered_results)}')
        #使用yoloface 把没有人脸的剔除掉
        '''
        for i in range(len(filtered_results) - 1, -1, -1):
            x1, y1, x2, y2, conf, class_id, person_img = filtered_results[i]
            results=self.yolo_face(person_img,conf=0.8)
            
            # 提取边界框和类别 ID
            hasFace=False
            # 检查 results 是否为列表
            if isinstance(results, list):
                for result in results:
                    # 提取边界框和类别 ID
                    boxes = result.boxes
                    class_ids = boxes.cls.cpu().numpy()  # 类别 ID
                    confidences = boxes.conf.cpu().numpy()  # 置信度
                    xyxy = boxes.xyxy.cpu().numpy()  # 边界框坐标 (x1, y1, x2, y2)

                    # 映射类别 ID 到类别名称
                    names = result.names
                    predicted_classes = [names[int(cls_id)] for cls_id in class_ids]

                    # 打印结果
                    for i, (cls, conf, bbox) in enumerate(zip(predicted_classes, confidences, xyxy)):
                        if conf>0.2:
                            hasFace=True
            else:
                print("results is not a list, but a single result object")
                # 提取边界框和类别 ID
                boxes = results.boxes
                class_ids = boxes.cls.cpu().numpy()  # 类别 ID
                confidences = boxes.conf.cpu().numpy()  # 置信度
                xyxy = boxes.xyxy.cpu().numpy()  # 边界框坐标 (x1, y1, x2, y2)

                # 映射类别 ID 到类别名称
                names = results.names
                predicted_classes = [names[int(cls_id)] for cls_id in class_ids]

                # 打印结果
                for i, (cls, conf, bbox) in enumerate(zip(predicted_classes, confidences, xyxy)):
                    if conf>0.2:
                            hasFace=True
            print(f"hasFace:{hasFace}")
            if not hasFace:
                filtered_results.pop(i)
        self.get_logger().info(f'有人脸的数量: {len(filtered_results)}')
        # 可选：绘制最终结果
        '''
        final_image = img.copy()
        for x1, y1, x2, y2, conf, class_id, _ in filtered_results:
            cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 使用绿色表示最终结果
        
        # 保存最终结果图像
        final_output_path = os.path.join(self.path, f'final_result_{self.classid}_{self.timer}_{timestamp}.jpg')
        cv2.imwrite(final_output_path, final_image)
        
        return filtered_results

    def _merge_overlapping_boxes(self, detections, img):
        """
        合并重叠的检测框
        
        Args:
            detections: 原始检测结果列表，每项为 (x1, y1, x2, y2, conf, class_id)
            img: 原始图像
            
        Returns:
            合并后的检测结果列表，每项为 (x1, y1, x2, y2, conf, class_id, person_img)
        """
        if not detections:
            return []

        # 按置信度排序
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        def calculate_iou(box1, box2):
            """计算两个框的IoU"""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0

        def boxes_distance(box1, box2):
            """计算两个框左上角的距离"""
            return ((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2) ** 0.5

        # 存储保留的检测框
        kept_detections = []
        used = set()

        # 遍历所有检测框
        for i, det1 in enumerate(detections):
            if i in used:
                continue

            current_group = [det1]
            used.add(i)

            # 查找与当前框重叠的其他框
            for j, det2 in enumerate(detections):
                if j in used or i == j:
                    continue

                iou = calculate_iou(det1[:4], det2[:4])
                distance = boxes_distance(det1, det2)

                # 如果IoU较大或距离较近，认为是同一个目标
                if iou > 0.3 or distance < 20:  # 可调整阈值
                    current_group.append(det2)
                    used.add(j)

            # 从当前组中选择最佳检测框
            if current_group:
                best_det = max(current_group, key=lambda x: x[4])  # 选择置信度最高的
                x1, y1, x2, y2, conf, class_id = best_det
                person_img = img[y1:y2, x1:x2]
                kept_detections.append((x1, y1, x2, y2, conf, class_id, person_img))
            
        
        return kept_detections

    def _is_valid_detection(self, box, img_shape):
        """
        检查检测框是否有效
        """
        x1, y1, x2, y2 = box[:4]
        height, width = img_shape[:2]
        
        # 检查坐标是否在图像范围内
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return False
            
        # 检查框的大小是否合理
        box_width = x2 - x1
        box_height = y2 - y1
        min_size = 30  # 最小尺寸阈值
        max_size = min(width, height) * 0.8  # 最大尺寸阈值
        
        if box_width < min_size or box_height < min_size:
            return False
        if box_width > max_size or box_height > max_size:
            return False
            
        # 检查宽高比是否合理
        aspect_ratio = box_width / box_height
        if aspect_ratio < 0.2 or aspect_ratio > 5:  # 可调整阈值
            return False
            
        return True

    def create_subscriptions(self):
        # 学生的相机信息处理
        for row in self.cameras:
            topic_name = f'camera_{row[0]}/compressed'
            
            
            if row[1] == 0:
                #subscriber = Subscriber(self, CompressedImage, topic_name)
                #self.subscribers.append(subscriber)

                subscription_tmp = self.create_subscription(
                    CompressedImage,
                    topic_name,
                    self.callback,
                    10)
                self.subscribers.append(subscription_tmp)
                self.get_logger().info(f'Subscrib stu to: {topic_name}')

            elif row[1] == 1:
                self.create_subscriptions_teacher(topic_name)
                self.get_logger().info(f'Subscribing tea to: {topic_name}')
        
        # Create a time synchronizer to synchronize messages from all cameras
        #self.ts = ApproximateTimeSynchronizer(self.subscribers, queue_size=10, slop=0.1)
        #self.ts.registerCallback(self.callback)

    def create_subscriptions_teacher(self,topic_name):
        #教师的相机信息处理
        # Create subscribers for each camera
        self.subscription_teacher = self.create_subscription(
            CompressedImage,
            topic_name,
            self.listener_callback_teacher,
            10)
        self.subscription_teacher  # 防止未使用的变量警告

    def listener_callback_teacher(self, msg):
        self.YOLOv7_teacher(self.bridge.compressed_imgmsg_to_cv2(msg))
    def callback(self, msg):
        self.get_logger().info('接收到消息')
        if self.indeximg==6:
            self.get_logger().info('开始检测')
            self.indeximg=0
        rgb_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        self.imgs.append(rgb_image)

    #def callback(self, *msgs):
    def taskImage(self, rgb_image):
        self.get_logger().info('接收到消息')
        if self.indeximg==6:
            self.get_logger().info('开始检测')
            self.indeximg=0
        try:
            # Process the synchronized messages from all cameras
            #在/data/pictures/(当前日期20250103)/下面保存图片,如果不存在目录就创建，
            

            # 检查图像是否有效
            if rgb_image is None or rgb_image.size == 0:
                self.get_logger().error("Received invalid image")
                return

            # 获取图像的形状
            shape = rgb_image.shape
            if shape[0] == 0 or shape[1] == 0:
                self.get_logger().error("Image has invalid dimensions")
                return
            # 检测人脸匹配在线状态。
            #self.MacFace(rgb_image)
            #能不能把rgb_image合成一张保存下来
            cv2.imwrite(f'{self.path}/rgb-{self.classid}-{self.timer}-{self.indeximg}.jpg', rgb_image)

            all_results=self.splitDect(rgb_image)
            for all_result in all_results:
                x1, y1, x2, y2, conf, class_id ,personimg= all_result
                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.imwrite(f'{self.path}/result_{datetime.datetime.now().strftime("%H:%M:%S")}.jpg', rgb_image)
            self.get_logger().info(f'检测到{len(all_results)}个人bbox')
            #把所有all_results里面的图片合并成一张大图
            #merged_image = self.merge_images(all_results)
            
            for result in all_results:
                x1, y1, x2, y2, conf, class_id, person_img = result
                results = self.yolo_stu(person_img, conf=0.4, iou=0.6, verbose=False)
                retType= self.findMaxAtResult(results)
                if retType is None:
                    continue
                self.class_counts[retType] += 1
                # if results[0].boxes.data.shape[0] > 0:
                #     # 获取置信度最高的结果
                #     best_result = results[0].boxes.data[results[0].boxes.data[:, 4].argmax()]
                #     x1, y1, x2, y2, conf, class_id = best_result
                #     # 只统计每个类别一次
                #     if self.class_counts[class_id] == 0:
                #         self.class_counts[class_id] += 1
                cv2.putText(rgb_image, f"{class_id:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #对分类进行统计，插入数据
            path=f'{self.path}/result4_{self.classid}_{datetime.datetime.now().strftime("%H:%M:%S")}.jpg'
            cv2.imwrite(path, rgb_image)
            self.class_img_path=path.replace('/data/', getIp(self.serverid))
            
            # 从所有检测结果中提取人的位置信息
            #循环检测结果，把图片切片的人脸信息传入人脸引擎，进行人脸识别
            for result in all_results:
                
                x1, y1, x2, y2, conf, class_id, person_img = result
                # 在这里处理每个人的位置信息
                #self.get_logger().info(f"Person detected at ({x1}, {y1}, {x2}, {y2}) with confidence {conf}")
                
                sim,max_sim_student=self.MacFace(person_img)
                if max_sim_student is not None:
                    cv2.putText(rgb_image, f"{sim:.2f}-{max_sim_student.sno}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            self.get_logger().info(f"classid-{self.classid}-{self.timer} 人脸处理完成") 
            path0=f'{self.path}/result3_{datetime.datetime.now().strftime("%H:%M:%S")}.jpg'
            cv2.imwrite(path0, rgb_image)
            

            #cv2.imwrite(f'{self.path}/end-{self.classid}-{self.timer}.jpg', rgb_image)
            
            
            # 发布检测结果,students里面的 studentObject转换成psersonMessage并发布出去
            # person_msgs = []
            # for student in self.students:
            #     person_msg = PersonMessage()
            #     person_msg.id = student.id
            #     person_msg.sno = student.sno
            #     person_msg.act = student.type
            #     person_msgs.append(person_msg)
            # persons=Persons()
            # persons.persons=person_msgs
            # self.pub_person.publish(persons)
            
            
        except Exception as e:
            self.get_logger().info(f"Error processing stu:classid- {self.classid}-{self.timer} images: {e}")
        finally:
            self.get_logger().info(f"classid-{self.classid}-{self.timer} images processed")
    def findMaxAtResult(self,results):
        
        # 初始化变量以存储最高置信度的检测结果
        highest_confidence = 0.0
        highest_confidence_class = None
        highest_confidence_bbox = None

        # 检查 results 是否为列表
        if isinstance(results, list):
            for result in results:
                # 提取边界框和类别 ID
                boxes = result.boxes
                class_ids = boxes.cls.cpu().numpy()  # 类别 ID
                confidences = boxes.conf.cpu().numpy()  # 置信度
                xyxy = boxes.xyxy.cpu().numpy()  # 边界框坐标 (x1, y1, x2, y2)

                # 映射类别 ID 到类别名称
                #names = result.names
                #predicted_classes = [names[int(cls_id)] for cls_id in class_ids]

                # 遍历所有检测结果，找到置信度最高的那个
                for cls, conf, bbox in zip(class_ids, confidences, xyxy):
                    if conf > highest_confidence:
                        highest_confidence = conf
                        highest_confidence_class = cls
                        highest_confidence_bbox = bbox
        else:
            # 提取边界框和类别 ID
            boxes = results.boxes
            class_ids = boxes.cls.cpu().numpy()  # 类别 ID
            confidences = boxes.conf.cpu().numpy()  # 置信度
            xyxy = boxes.xyxy.cpu().numpy()  # 边界框坐标 (x1, y1, x2, y2)

            # 映射类别 ID 到类别名称
            #names = results.names
            #predicted_classes =class_ids# [names[int(cls_id)] for cls_id in class_ids]

            # 遍历所有检测结果，找到置信度最高的那个
            for cls, conf, bbox in zip(class_ids, confidences, xyxy):
                if conf > highest_confidence:
                    highest_confidence = conf
                    highest_confidence_class = cls
                    highest_confidence_bbox = bbox

        # 打印置信度最高的检测结果
        print(f"最高置信度的检测结果：")
        print(f"  类别: {highest_confidence_class}")
        print(f"  置信度: {highest_confidence:.2f}")
        print(f"  边界框: {highest_confidence_bbox}")
        return highest_confidence_class 
    #出勤率检测，self.students是所有学生的信息
    def MacFace(self,rgb_image):
        sim=0
        max_sim_student = None
        try:
            featur1=self.face_engine.detect_and_get_face_featureOne(rgb_image)
            if featur1 is None:
                self.get_logger().info('no face detected')
                return 0
            max_sim = 0
            
            #print('打印featur1')
            #self.face_engine.printFeature(featur1)
            
            for student in self.students:
                featur2 = student.feature
                #print('打印featur2')
                #self.face_engine.printFeature(featur2)
                sim = self.face_engine.compare_faces(featur1, featur2)
                #self.get_logger().info(f'比对结果：{sim}')
                if sim > max_sim:
                    max_sim = sim
                    max_sim_student = student

            if max_sim > 0.25 and max_sim_student is not None:
                # 找到匹配的学生，更新其状态
                max_sim_student.lasttime = self.timer
                max_sim_student.online = 1
                self.get_logger().info(f'匹配到学生-{max_sim}:{max_sim_student.sno}')
                # 把结果绘制到图片上
                #cv2.putText(rgb_image, f"s: {max_sim:.2f}-type:{self.class_id}", (0, -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
        except Exception as e:
            self.get_logger().info(f"Error processing stu at arcsoft:classid- {self.classid}-{self.timer} images: {e}")
        finally:
            return sim,max_sim_student

    def testpublic(self):
        self.get_logger().info('testpublic')
        
        person_msg = PersonMessage()
        person_msg.id = 1
        person_msg.sno = '0011'
        person_msg.act = 1
        persons=Persons()
        persons.persons=person_msg
        self.publisher.publish(persons)    

    def merge_images(self,all_results):
        # 计算大图的尺寸
        max_x = max([result[2] for result in all_results])
        max_y = max([result[3] for result in all_results])

        # 创建一个空白的大图
        merged_image = np.zeros((max_y, max_x, 3), dtype=np.uint8)

        # 将每个小图放置在大图的适当位置
        for result in all_results:
            x1, y1, x2, y2, _, _, person_img = result
            merged_image[y1:y2, x1:x2] = person_img

        return merged_image
      
    

    def get_camera_list_by_classid(self, classid):
        """
        根据classtable的id，从三个表中取出摄像头的列表。
        """
        try:
            # 连接到数据库
            cursor=self.conn.cursor()
            # 查询classroom表，获取camera_id
            cursor.execute("SELECT id,type FROM cameras  WHERE type<2 and id IN ( select camera_id from classroom_camera where classroom_id in (select classid_id from classtable where id=%s))", (self.classid,))
            camera_list =[]
            for row3 in cursor.fetchall():
                camera_list.append(row3)
            self.get_logger().info(f'Camera list for classid {classid}: {camera_list}')
            return camera_list

        except Exception as e:
            self.get_logger().error(f'Error retrieving camera list for classid {classid}: {e}')
            return []
    
    def YOLOv7_teacher(self,img):
        """
        教师的相机信息处理
        """
        try:
           # img = self.GetTorchImg(img)
            results = self.yolo_teacher(img, conf=0.3, iou=0.5)
            best_match = None
            best_conf = -1
            best_class_id=-1
            teachsno=''
            path2=''
            if hasattr(self, 'teachers') and len(self.teachers) > 0:  # 使用 len() 检查列表长度
                teachsno = self.teachers[0].sno
                self.get_logger().info(f'Found teacher sno: {teachsno}')
            else:
                self.get_logger().info('No teacher found')

            self.get_logger().info(f'teacher yolo get img')
            for result in results[0].boxes.data.clone():  # 克隆张量以避免原地更新错误
                x1, y1, x2, y2, conf, class_id = result
                conf_value = float(conf) if torch.is_tensor(conf) else conf.item() if isinstance(conf, np.ndarray) else conf
                if conf_value > best_conf:
                    best_conf = conf_value
                    best_match = result
            
            self.get_logger().info(f'Best teach match: {best_match}')

            if best_match is not None:
                x1, y1, x2, y2, conf, class_id = best_match
                best_class_id = int(class_id.item()) if torch.is_tensor(class_id) else int(class_id)
                path2=f'{self.path}/teacher-{self.classid}-{self.timer}.jpg'
                cv2.imwrite(path2, img)
                # 在这里处理最佳匹配的结果
                
            else:
                # 没有找到匹配的结果
                
                pass
            #提交到数据库
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO aiclass (sno,ai,time,classid,time2,img) VALUES (%s,%s,%s, %s,%s,%s)", \
                        (teachsno,best_class_id,self.timer,self.classid,\
                         (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),\
                            path2.replace('/data/',getIp(self.serverid))))
            self.conn.commit()  # 提交事务
            cursor.close()
        except Exception as e:
            self.get_logger().error(f"Error processing teacher:classid- {self.classid}-{self.timer} images: {e}")

    def GetTorchImg(self,img):
        img = cv2.flip(cv2.flip(np.asanyarray(img), 0), 1)  # Camera is upside down on the Go1
        im0 = img.copy()
        img = img[np.newaxis, :, :, :]
        img = np.stack(img, 0)
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return im0  

    

    def arcsoft_face_recognition(self,face_image):


        # 示例代码：假设ArcSoft接口已经导入并初始化
        #

        # 初始化ArcSoft人脸识别引擎
        # = FaceEngine()

        # 进行人脸检测
        face_detection_result = self.face_engine.detect_faces(face_image)

        # 如果检测到人脸
        if face_detection_result is not None and len(face_detection_result) > 0:
            # 提取人脸特征
            face_features = self.face_engine.extract_face_features(face_image, face_detection_result[0])

            # 返回人脸特征向量
            return face_features

        # 如果未检测到人脸，返回None
        return None
    
    def _cleanup(self):
        """清理资源的函数"""
        try:
            # 获取当前进程
            current_process = psutil.Process(self.pid)
            
            # 获取所有子进程
            children = current_process.children(recursive=True)
            
            # 终止所有子进程
            for child in children:
                try:
                    child.terminate()
                except:
                    pass
            
            # 确保资源被释放
            self.destroy_node()
            
            self.get_logger().info('Cleanup completed')
            
        except Exception as e:
            self.get_logger().error(f'Error during cleanup: {str(e)}')
    

def main(args=None):
    
    rclpy.init(args=args)
    with torch.no_grad():
        node = ObjectDetection()
        rclpy.spin(node)
        if node.conn is not None:
            node.conn.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
