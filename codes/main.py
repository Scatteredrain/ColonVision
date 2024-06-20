from UIFunctions import *
from utils_seg_class.classification import ClsHead as BaseClshead
from utils_seg_class.segment import segmentation as BaseSegmenter
from AIDetector_pytorch import Detector as BaseDetector
import numpy as np
import cv2
import os
import json
import sys
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from PySide6.QtCore import QObject,Signal,QTimer
from PySide6.QtWidgets import QMainWindow

class Segmenter(BaseSegmenter,BaseClshead, QObject, BaseDetector):

    seg2main_pre_img = Signal(np.ndarray)   # raw image signal
    seg2main_res_img = Signal(np.ndarray)   # test result signal
    seg2main_status_msg = Signal(str)       # Detecting/pausing/stopping/testing complete/error reporting signal
    seg2main_target_num = Signal(str)       # Targets detected
    seg2main_class_num = Signal(str)        # conf
    update_frame_signal = Signal()


    def __init__(self):
        super(Segmenter, self).__init__()

        # GUI args

        self.source = ''                  # input source
        self.stop_dtc = False             # Termination detection
        self.continue_dtc = True          # pause
        self.save_res = False             # Save test results
        self.save_txt = False             # save label(txt) file
        self.save_path = './results_save' # save path
        os.makedirs(self.save_path,exist_ok=True)
        self.flag = 0
        self.mod = ''
        self.speed_thres = 10

              
        self.timer = QTimer(self)
        self.update_frame_signal.connect(self.update_frame)

    def save_preds(self, img, filename):
        cv2.imwrite(filename, img)

     # main for video-detect
    def on_timeout(self):
        self.update_frame_signal.emit()

    def update_frame(self):

        # while cap.isOpened():
        if self.continue_dtc:
            ret, img = self.cap.read()
            if ret:
                img0 = img.copy()
                result = self.feedCap(img)
                # result = det.detect(im)[0]
                pred = result['frame']

                # cls = result['clss'][0] 
                # if cls == 1:
                #     label = "腺瘤型"
                # else:
                #     label = "增生型"
                # conf =  str('%.2f'%(result['confs'][0]*100-15.34)+'%')
                self.seg2main_res_img.emit(pred)
                self.seg2main_pre_img.emit(img0)

                # self.seg2main_target_num.emit(label)
                # self.seg2main_class_num.emit(conf)
                # cv2.waitKey(int(1000/60)) # 60 fps
                # time.sleep(self.speed_thres/1000)
            else:
                self.timer.stop()
        if self.stop_dtc:
            self.timer.stop()

    def run(self):
        try:
            if self.flag == 0:
                self.seg2main_status_msg.emit('Loding Model...')
                self.load_model()
                self.load_model_cls()
                self.init_model()
                self.seg2main_status_msg.emit('Done Model Loding!')
                self.flag = 1
            else:
                self.seg2main_status_msg.emit('Model Completed')


################# model working
################# Send test results    
            if self.continue_dtc:
                # time.sleep(0.001)
                self.seg2main_status_msg.emit('Processing...')
                file = self.source

                if file.endswith('.jpg') or file.endswith('.png'):
                    im0,im0s,img_cls = self.segment(file)
                    pred,conf = self.detectpolyp(img_cls)

                    if pred == 1:
                        label = "腺瘤型"
                    else:
                        label = "增生型"
                    conf = str('%.2f'%(conf*100-15.34)+'%')

                    self.seg2main_res_img.emit(im0s) # after detection
                    print("img show")
                    self.seg2main_pre_img.emit(im0)   # Before testing

                    self.seg2main_target_num.emit(label)
                    self.seg2main_class_num.emit(conf)

                elif file.endswith('.mp4'): 
                    self.cap = cv2.VideoCapture(file)

                    self.timer.timeout.connect(self.on_timeout)
                    self.timer.start(33)  # 视频帧率大约是30fps
              
                # save img result
                if self.save_res:
                    name = os.path.basename(self.source)
                    self.save_preds(im0s,self.save_path + '/' + name)
                    print('save: '+self.source)

                # save label result
                if self.save_txt:
                    txt = self.save_path + '/' +name[:-3] +'txt'
                    with open(txt, 'a') as f:
                        f.write('\n图片：{}，类别：{}，置信度：{}'.format(name,label,conf))



            # Detection completed
            self.seg2main_status_msg.emit('Finish processing.')

        except Exception as e:
            pass
            print(e)
            self.seg2main_status_msg.emit('%s' % e)


class MainWindow(QMainWindow, Ui_MainWindow):
    main2seg_begin_sgl = Signal()  # The main window sends an execution signal to the model instance
    def __init__(self, parent=None): 
        super(MainWindow, self).__init__(parent)
        # basic interface
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # rounded transparent
        self.setWindowFlags(Qt.FramelessWindowHint)  # Set window flag: hide window borders
        UIFuncitons.uiDefinitions(self)

        # Show module shadows
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(101, 157, 139))



        # thread
        self.seg_predict = Segmenter()                           # Create a seg instance

        self.seg_thread = QThread()                                  # Create seg thread
        self.seg_predict.seg2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video))
        self.seg_predict.seg2main_res_img.connect(lambda x: self.show_image(x, self.res_video))
        self.seg_predict.seg2main_status_msg.connect(lambda x: self.show_status(x))

        self.seg_predict.seg2main_target_num.connect(lambda x:self.Target_num.setText(str(x)))
        self.seg_predict.seg2main_class_num.connect(lambda x: self.Class_num.setText(str(x)))
        self.main2seg_begin_sgl.connect(self.seg_predict.run)
        self.seg_predict.moveToThread(self.seg_thread)

        # Prompt window initialization
        self.Target_num.setText('--')
        self.Class_num.setText('--')
        
        # Select detection source
        self.src_file_button.clicked.connect(self.open_src_file)  # select local file
        # self.src_cam_button.clicked.connect(self.show_status("The function has not yet been implemented."))#chose_cam
        # self.src_rtsp_button.clicked.connect(self.show_status("The function has not yet been implemented."))#chose_rtsp

        # start testing button
        self.run_button.clicked.connect(self.run_or_continue)   # pause/start
        self.stop_button.clicked.connect(self.stop)             # termination

        # Other function buttons
        self.save_res_button.toggled.connect(self.is_save_res)  # save image option
        self.save_txt_button.toggled.connect(self.is_save_txt)  # Save label option
        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))   # left navigation button
        #self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))   # top right settings button


    # The main window displays the original image and detection results
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep the original data ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # Control start/pause
    def run_or_continue(self):
        if self.seg_predict.source == '':
            self.show_status('Please select a video source before starting...')
            self.run_button.setChecked(False)
        else:
            self.seg_predict.stop_dtc = False
            if self.run_button.isChecked():
                self.run_button.setChecked(True)    # start button
                self.save_txt_button.setEnabled(False)  # It is forbidden to check and save after starting the detection
                self.save_res_button.setEnabled(False)
                self.show_status('Processing...')
                self.seg_predict.continue_dtc = True   # Control whether Yolo is paused
                if not self.seg_thread.isRunning():
                    self.seg_thread.start()
                    self.main2seg_begin_sgl.emit()   # 释放运行信号

            else:
                self.seg_predict.continue_dtc = False
                self.show_status("Pause...")
                self.run_button.setChecked(False)    # start button

    # bottom status bar information
    def show_status(self, msg):
        self.status_bar.setText(msg)
        if msg == 'Finish processing!' or msg == '检测完成':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)    
            # self.progress_bar.setValue(0)
            if self.seg_thread.isRunning():
                self.seg_thread.quit()         # end process
        elif msg == 'Process terminated!' or msg == '检测终止':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)    
            # self.progress_bar.setValue(0)
            if self.seg_thread.isRunning():
                self.seg_thread.quit()         # end process
            self.pre_video.clear()           # clear image display  
            self.res_video.clear()
            self.Target_num.setText('--')
            self.Class_num.setText('--')


    # select local file
    def open_src_file(self):
        config_file = 'config/fold.json'    
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']     
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.seg_predict.source = name
            self.show_status('Load File：{}'.format(os.path.basename(name))) 
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)  
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()             


    # Save test result button--picture/video
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Run image results are not saved.')
            self.seg_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Run image results will be saved.')
            self.seg_predict.save_res = True
    
    # Save test result button -- label (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Labels results are not saved.')
            self.seg_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Labels results will be saved.')
            self.seg_predict.save_txt = True

    # Configuration initialization  ~~~wait to change~~~
    def load_config(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            save_res = 0   
            save_txt = 0    
            new_config = {
                          "save_res": save_res,
                          "save_txt": save_txt
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:

                save_res = 0
                save_txt = 0
            else:

                save_res = config['save_res']
                save_txt = config['save_txt']
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.seg_predict.save_res = (False if save_res == 0 else True)
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt)) 
        self.seg_predict.save_txt = (False if save_txt == 0 else True)
        self.run_button.setChecked(False)  
        self.show_status("Welcome~")

    # Terminate button and associated state
    def stop(self):
        if self.seg_thread.isRunning():
            self.seg_thread.quit()         # end thread
        self.seg_predict.stop_dtc = True
        self.run_button.setChecked(False)    # start key recovery
        self.save_res_button.setEnabled(True)   # Ability to use the save button
        self.save_txt_button.setEnabled(True)   # Ability to use the save button
        self.pre_video.clear()           # clear image display
        self.res_video.clear()           # clear image display
        self.Target_num.setText('--')
        self.Class_num.setText('--')
 




    # Get the mouse position (used to hold down the title bar and drag the window)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize the adjustment when dragging the bottom and right edges of the window size
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)

    # Exit  thread, save settings
    def closeEvent(self, event):
        config_file = 'config/setting.json'
        config = dict()
        config['save_res'] = (0 if self.save_res_button.checkState()==Qt.Unchecked else 2)
        config['save_txt'] = (0 if self.save_txt_button.checkState()==Qt.Unchecked else 2)
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        # Exit the process before closing
        if self.seg_thread.isRunning():
            self.seg_predict.stop_dtc = True
            self.seg_thread.quit()
            MessageBox(
                self.close_button, title='Note', text='Exiting, please wait...', time=3000, auto=True).exec()
            sys.exit(0)
        else:
            sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())  
