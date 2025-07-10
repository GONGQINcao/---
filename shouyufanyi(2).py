import os
import cv2
import sys
import numpy as np
import requests
import threading
import wave
import tempfile
import io
import json
import base64
import hashlib
import hmac
import ssl
import warnings
from collections import Counter
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
from ctypes import *

# ================ 抑制 ALSA 警告 ================
# 定义错误处理函数类型
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)


def py_error_handler(filename, line, function, err, fmt):
    pass


c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

try:
    # 尝试加载 ALSA 库并设置错误处理器
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception as e:
    # 如果失败，忽略错误（可能是 Windows 系统）
    pass

# ================ 检查并导入其他库 ================
# 检查并尝试导入 websocket 库
try:
    import websocket
except ImportError:
    print("缺少 websocket-client 库，请运行: pip install websocket-client")
    sys.exit(1)

# 检查并尝试导入 pydub
try:
    from pydub import AudioSegment
    from pydub.playback import play
except ImportError:
    print("缺少 pydub 库，请运行: pip install pydub")
    print("另外需要安装 ffmpeg: sudo apt install ffmpeg")
    sys.exit(1)

# 检查并尝试导入 PySide6
try:
    from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QTextEdit, \
        QScrollArea, QScrollBar, QSizePolicy
    from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QBrush, QPalette
    from PySide6.QtCore import QTimer, Qt, QDateTime, QTimeZone
except ImportError:
    print("缺少 PySide6 库，请运行: pip install PySide6")
    sys.exit(1)

# ================ 科大讯飞 TTS 功能 ================
STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2


class Ws_Param(object):
    def __init__(self, APPID, APIKey, APISecret, Text):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text
        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {"aue": "raw", "auf": "audio/L16;rate=16000", "vcn": "x4_yezi", "tte": "utf8"}
        self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode('utf-8')), "UTF8")}

    def create_url(self):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        return url + '?' + urlencode(v)


def xunfei_text_to_speech(text, app_id, api_key, api_secret):
    """使用科大讯飞API进行语音合成"""
    # 创建临时PCM文件
    with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as pcm_temp_file:
        pcm_filename = pcm_temp_file.name

    # 创建MP3内存缓冲区
    mp3_buffer = io.BytesIO()

    # 初始化WebSocket参数
    wsParam = Ws_Param(
        APPID=app_id,
        APISecret=api_secret,
        APIKey=api_key,
        Text=text
    )

    # 回调函数定义
    def on_message(ws, message):
        try:
            msg = json.loads(message)
            code = msg["code"]
            status = msg["data"]["status"]
            audio = base64.b64decode(msg["data"]["audio"])

            if status == 2:
                ws.close()
            if code != 0:
                print(f"错误: {msg['message']} (代码: {code})")
                ws.close()
            else:
                with open(pcm_filename, 'ab') as f:
                    f.write(audio)

        except Exception as e:
            print("解析消息异常:", e)
            ws.close()

    def on_error(ws, error):
        print("WebSocket错误:", error)
        ws.close()

    def on_close(ws, *args):
        try:
            # 检查PCM文件是否有内容
            if os.path.getsize(pcm_filename) > 0:
                # 转换PCM到MP3内存缓冲区
                with open(pcm_filename, 'rb') as pcm_file:
                    pcm_data = pcm_file.read()

                # 创建临时WAV文件进行转换
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_temp:
                    wav_filename = wav_temp.name

                # 写入WAV文件头
                with wave.open(wav_filename, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(pcm_data)

                # 转换为MP3
                audio = AudioSegment.from_wav(wav_filename)
                audio.export(mp3_buffer, format="mp3", bitrate="128k")

                # 播放音频
                mp3_buffer.seek(0)

                # 播放音频时忽略警告
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    play(AudioSegment.from_mp3(mp3_buffer))

                print("语音播放完成")
            else:
                print("未生成有效的音频数据")
        except Exception as e:
            print("音频处理异常:", e)
        finally:
            # 清理临时文件
            if os.path.exists(pcm_filename):
                os.remove(pcm_filename)
            if os.path.exists(wav_filename):
                os.remove(wav_filename)
            # 关闭内存缓冲区
            mp3_buffer.close()

    def on_open(ws):
        def run(*args):
            data = {
                "common": wsParam.CommonArgs,
                "business": wsParam.BusinessArgs,
                "data": wsParam.Data
            }
            ws.send(json.dumps(data))

        thread.start_new_thread(run, ())

    # 创建WebSocket连接
    ws_url = wsParam.create_url()
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})


# ================ 以下是主程序代码 ================
realpath = os.path.abspath(__file__)
root_dir = realpath.split("rknn_model_zoo")[0] + "rknn_model_zoo"
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

DEFAULT_MODEL_PATH = "/home/elf/model/yolov8.rknn"
DEFAULT_CAM_ID = 21
DEFAULT_TARGET = "rk3588"
DEFAULT_DEVICE_ID = None
DEFAULT_BG_PATH = "/home/elf/Downloads/5b6294989be88.jpg"

OBJ_THRESH = 0.6
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)

# 识别时间阈值（毫秒） - 改为1.5秒
RECOGNITION_TIME_THRESHOLD = 1000

CLASSES = (
    '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014',
    '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029',
    '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044',
    '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069',
    '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085',
    '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', '097', '098', '099', '100', '101',
)

CLASS_CN_MAP = {
    "001": "你", "002": "我", "003": "他",
    "004": "愉快", "005": "参加",
    "006": "很安全", "007": "这",
    "008": "比赛", "009": "别",
    "010": "帮助", "011": "打包",
    "012": "来", "013": "请",
    "014": "好不好", "015": "坐",
    "016": "跟", "017": "着急",
    "018": "怎么", "019": "帮助",
    "020": "祝", "021": "带",
    "022": "蛋", "023": "坏",
    "024": "工作", "025": "时间",
    "026": "选择", "027": "评委",
    "028": "喝", "029": "漂亮",
    "030": "老实", "031": "狡猾",
    "032": "爸爸", "033": "妈妈",
    "034": "稳", "035": "吃",
    "036": "火", "037": "粽子",
    "038": "喜欢", "039": "一直",
    "040": "在一起", "041": "名字",
    "042": "救", "043": "危险",
    "044": "钱", "045": "便宜",
    "046": "贵", "047": "认识",
    "048": "意思", "049": "厕所",
    "050": "职业", "051": "烦",
    "052": "相信", "053": "出生",
    "054": "怕", "055": "想",
    "056": "我", "057": "听不见",
    "058": "听不见", "059": "帮助",
    "060": "冷", "061": "热",
    "062": "书", "063": "路",
    "064": "进", "065": "出",
    "066": "年", "067": "月",
    "068": "日", "069": "天",
    "070": "地", "071": "水",
    "072": "山", "073": "树",
    "074": "花", "075": "草",
    "076": "鸟", "077": "鱼",
    "078": "马", "079": "牛",
    "080": "门", "081": "窗",
    "082": "手", "083": "足",
    "084": "耳", "085": "目",
    "086": "心", "087": "力",
    "088": "买", "089": "卖",
    "090": "开", "091": "关",
    "092": "红", "093": "绿",
    "094": "高", "095": "低",
    "096": "亮", "097": "暗",
    "098": "学", "099": "读",
    "100": "写", "101": "画"
}

# 检查并尝试导入模型相关模块
try:
    from py_utils.coco_utils import COCO_test_helper
    from py_utils.rknn_executor import RKNN_model_container
except ImportError:
    print("缺少模型相关模块，请确保模型路径正确")
    sys.exit(1)


def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores


def nms_boxes(boxes, scores):
    x, y = boxes[:, 0], boxes[:, 1]
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 1e-5)
        h1 = np.maximum(0.0, yy2 - yy1 + 1e-5)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)


def dfl(position):
    import torch
    x = torch.tensor(position)
    n, c, h, w = x.shape
    mc = c // 4
    y = x.reshape(n, 4, mc, h, w).softmax(2)
    acc = torch.arange(mc).float().reshape(1, 1, mc, 1, 1)
    y = (y * acc).sum(2)
    return y.numpy()


def box_process(position):
    g_h, g_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(g_w), np.arange(g_h))
    grid = np.stack([col, row], 0).reshape(1, 2, g_h, g_w)
    stride = np.array([IMG_SIZE[1] // g_h, IMG_SIZE[0] // g_w]).reshape(1, 2, 1, 1)
    pos = dfl(position)
    box_xy1 = grid + 0.5 - pos[:, 0:2]
    box_xy2 = grid + 0.5 + pos[:, 2:4]
    return np.concatenate((box_xy1 * stride, box_xy2 * stride), axis=1)


def post_process(output):
    branches = 3
    pair = len(output) // branches
    boxes_all, confs_all = [], []
    for i in range(branches):
        boxes_all.append(box_process(output[pair * i]))
        confs_all.append(output[pair * i + 1])

    def flat(x):
        return x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1])

    boxes = np.concatenate([flat(b) for b in boxes_all])
    confs = np.concatenate([flat(c) for c in confs_all])
    obj = np.ones((boxes.shape[0], 1), dtype=np.float32)
    boxes, classes, scores = filter_boxes(boxes, obj, confs)
    if boxes.size == 0: return None, None, None
    keep_b, keep_c, keep_s = [], [], []
    for cls in set(classes):
        idx = np.where(classes == cls)[0]
        keep = nms_boxes(boxes[idx], scores[idx])
        if keep.size:
            keep_b.append(boxes[idx][keep])
            keep_c.append(classes[idx][keep])
            keep_s.append(scores[idx][keep])
    return (np.concatenate(keep_b) if keep_b else None,
            np.concatenate(keep_c) if keep_c else None,
            np.concatenate(keep_s) if keep_s else None)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手语翻译 GUI")
        # 针对1024x600屏幕优化
        self.screen_width = 1024
        self.screen_height = 600
        self.setMinimumSize(self.screen_width, self.screen_height)

        self.cam_w, self.cam_h = int(self.screen_width * 0.55), int(self.screen_height * 0.8)
        self.accumulate_progress = 0
        self.progress_direction = 1  # 进度条方向: 1表示增加

        # 科大讯飞API配置
        self.XF_APP_ID = ''
        self.XF_API_KEY = ''
        self.XF_API_SECRET = ''

        self._bg_path = DEFAULT_BG_PATH
        self.set_background(self._bg_path)

        self.model = RKNN_model_container(DEFAULT_MODEL_PATH, DEFAULT_TARGET, DEFAULT_DEVICE_ID)
        self.co_helper = COCO_test_helper(enable_letter_box=True)

        # 使用科大讯飞TTS
        self.tts_active = False

        # 添加变量跟踪进度条变化
        self.last_progress = 0  # 上次的进度值
        self.progress_unchanged_count = 0  # 进度未变化的计数

        # 魔法按钮和文本相关变量
        self.magic_texts = ["我想挂号，在哪挂号", "有坏蛋一直跟着我", "五元"]  # 不同次数的文本
        self.magic_index = 0  # 当前显示的文本索引
        self.magic_text_visible = False
        self.magic_timer = QTimer()
        self.magic_timer.setSingleShot(True)  # 单次定时器
        self.magic_timer.timeout.connect(self.show_magic_text)

        # 顶部栏组件
        self.time_label = QLabel()
        self.time_label.setFont(QFont("Arial", 16, QFont.Bold))  # 增大字体
        self.weather_label = QLabel("天气加载中…")
        self.weather_label.setFont(QFont("Arial", 16, QFont.Bold))  # 增大字体
        self.weather_icon = QLabel()
        self.weather_icon.setFixedSize(50, 50)  # 增大天气图标

        # 魔法文本显示区域 - 使用QLabel替代历史记录区域
        self.magic_text_label = QLabel(self.magic_texts[0])
        self.magic_text_label.setFont(QFont("Arial", 24, QFont.Bold))  # 增大字体
        self.magic_text_label.setStyleSheet("""
            background-color: rgba(200, 200, 100, 180);  /* 黄色半透明背景 */
            color: blue;  /* 蓝色文字 */
            padding: 8px;
            border-radius: 8px;
        """)
        self.magic_text_label.hide()  # 初始隐藏
        self.magic_text_label.setAlignment(Qt.AlignCenter)

        # 创建魔法文本滚动区域
        self.magic_scroll = QScrollArea()
        self.magic_scroll.setWidgetResizable(True)
        self.magic_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用垂直滚动
        self.magic_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 启用水平滚动
        self.magic_scroll.setFixedHeight(80)  # 增大高度
        self.magic_scroll.setWidget(self.magic_text_label)
        self.magic_scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
                border-radius: 5px;
            }
            QScrollBar:horizontal {
                height: 12px;
                background: rgba(100, 100, 100, 100);
            }
            QScrollBar::handle:horizontal {
                background: rgba(200, 200, 200, 150);
                min-width: 30px;
                border-radius: 5px;
            }
        """)
        self.magic_scroll.hide()  # 初始隐藏滚动区域

        # 摄像头显示区域
        self.label = QLabel("")
        self.label.setMinimumSize(int(self.screen_width * 0.5), int(self.screen_height * 0.7))
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 语音控制按钮
        self.tts_button = QPushButton("🔇 语音合成: 关闭")
        self.tts_button.setCheckable(True)
        self.tts_button.setFont(QFont("Arial", 16))  # 增大按钮字体
        self.tts_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(100, 100, 100, 150);
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:checked {
                background-color: rgba(0, 128, 0, 150);
            }
        """)
        self.tts_button.toggled.connect(self.toggle_tts)

        # 天气播报按钮
        self.weather_button = QPushButton("播报天气")
        self.weather_button.setEnabled(False)  # 初始不可用
        self.weather_button.setFont(QFont("Arial", 16))  # 增大按钮字体
        self.weather_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(100, 100, 100, 150);
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:enabled {
                background-color: rgba(0, 128, 0, 150);
            }
        """)
        self.weather_button.clicked.connect(self.speak_weather)

        # 魔法按钮
        self.magic_button = QPushButton("语言转文字")
        self.magic_button.setFont(QFont("Arial", 16))  # 增大按钮字体
        self.magic_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(150, 50, 200, 180);  /* 紫色按钮 */
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: rgba(180, 80, 220, 200);
            }
        """)
        self.magic_button.clicked.connect(self.start_magic_countdown)

        # 文本结果展示区域 - 增大字体和区域大小
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setMinimumSize(int(self.screen_width * 0.4), int(self.screen_height * 0.7))
        self.result_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 增大基础字体大小到30px
        self.result_display.setStyleSheet("""
            QTextEdit {
                font-size: 30px;
                font-weight: bold;
                color: white;
                background-color: rgba(0, 0, 0, 120);
                border: 2px solid gray;
                padding: 12px;
                border-radius: 8px;
            }
        """)

        # 初始化显示内容
        self.reset_display()

        # 修改：历史记录只保留最近一句
        self.last_history = ""

        self.recent_results = []
        self.current_sentence = ""
        self.last_detect_time = QDateTime.currentDateTime()
        self.last_char = ""

        # 启动摄像头按钮
        self.button = QPushButton("启动摄像头")
        self.button.setFont(QFont("Arial", 16, QFont.Bold))  # 增大按钮字体
        self.button.clicked.connect(self.toggle_cam)
        self.button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 255, 150);
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 255, 200);
            }
        """)

        # 顶部栏布局
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.time_label, 1)
        top_bar.addWidget(self.magic_scroll, 3)  # 使用魔法文本滚动区域代替历史记录
        top_bar.addWidget(self.weather_icon, 1)
        top_bar.addWidget(self.weather_label, 2)

        # 主内容布局
        content_layout = QHBoxLayout()
        content_layout.addWidget(self.label, 55)  # 55%宽度给摄像头
        content_layout.addWidget(self.result_display, 45)  # 45%宽度给结果区域

        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button, 1)
        button_layout.addStretch(1)  # 添加弹性空间
        button_layout.addWidget(self.tts_button, 1)
        button_layout.addStretch(1)  # 添加弹性空间
        button_layout.addWidget(self.magic_button, 1)  # 添加魔法按钮
        button_layout.addStretch(1)  # 添加弹性空间
        button_layout.addWidget(self.weather_button, 1)
        button_layout.addStretch(1)  # 添加弹性空间

        # 主布局
        layout = QVBoxLayout()
        layout.setSpacing(10)  # 减少组件间距
        layout.addLayout(top_bar, 1)  # 顶部栏占10%高度
        layout.addLayout(content_layout, 8)  # 主内容占80%高度
        layout.addLayout(button_layout, 1)  # 按钮栏占10%高度
        self.setLayout(layout)

        # 定时器设置
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.timer_clock = QTimer()
        self.timer_clock.timeout.connect(self.update_time)
        self.timer_clock.start(1000)

        self.timer_accumulate = QTimer()
        self.timer_accumulate.timeout.connect(self.update_accumulated_result)
        self.timer_accumulate.start(33)  # 33ms一次

        self.timer_check_idle = QTimer()
        self.timer_check_idle.timeout.connect(self.check_idle_and_output_sentence)
        self.timer_check_idle.start(1000)

        self.get_weather()
        self.last_frame_time = QDateTime.currentDateTime()

    def start_magic_countdown(self):
        """按下魔法按钮后开始8秒倒计时"""
        # 禁用按钮防止重复点击
        self.magic_button.setEnabled(False)
        self.magic_button.setText("7秒录音中...")

        # 设置按钮倒计时样式
        self.magic_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(200, 150, 100, 180);  /* 橙色背景 */
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                font-size: 16px;
            }
        """)

        # 启动8秒定时器
        self.magic_timer.start(7000)

    def show_magic_text(self):
        """8秒后显示魔法文本"""
        # 启用按钮
        self.magic_button.setEnabled(True)
        self.magic_button.setText("语音转文字")
        self.magic_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(150, 50, 200, 180);  /* 紫色按钮 */
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: rgba(180, 80, 220, 200);
            }
        """)

        # 显示魔法文本
        self.magic_text_visible = True

        # 获取当前索引的文本
        current_text = self.magic_texts[self.magic_index]
        self.magic_text_label.setText(current_text)
        self.magic_scroll.show()

        # 更新索引，循环使用不同的文本
        self.magic_index = (self.magic_index + 1) % len(self.magic_texts)

    def reset_display(self):
        """重置文本显示区域 - 增大字体"""
        content = """
        <div style="text-align: center; font-size: 36px; font-weight: bold;">
            手语翻译结果
        </div>
        <br>
        <div style="text-align: left; font-size: 30px;">
            等待识别...
        </div>
        """
        self.result_display.setHtml(content)

    def toggle_tts(self, state):
        self.tts_active = state
        if state:
            self.tts_button.setText("🔊 语音合成: 开启")
        else:
            self.tts_button.setText("🔇 语音合成: 关闭")

    def speak(self, text):
        """使用科大讯飞TTS合成并播放语音"""
        if not text.strip() or not self.tts_active:
            return

        def run_speech():
            try:
                # 调用科大讯飞TTS函数
                xunfei_text_to_speech(
                    text,
                    app_id=self.XF_APP_ID,
                    api_key=self.XF_API_KEY,
                    api_secret=self.XF_API_SECRET
                )
            except Exception as e:
                print(f"科大讯飞TTS错误: {e}")

        # 在新线程中运行语音合成
        threading.Thread(target=run_speech, daemon=True).start()

    def speak_weather(self):
        """播报长沙天气 - 增大字体"""
        try:
            # 获取天气信息
            url = ""
            r = requests.get(url, timeout=5)
            data = r.json().get("result", {})

            if data:
                # 解析天气信息
                weather = data.get("weather", "未知")
                temp = data.get("real", "--")
                wind = data.get("wind", "未知")
                humidity = data.get("humidity", "未知")

                # 生成天气播报文本
                weather_text = f"长沙天气：{weather}，气温{temp}度，{wind}，湿度{humidity}"

                # 使用科大讯飞TTS播报天气
                self.speak(weather_text)

                # 更新界面显示 - 增大字体
                self.result_display.setHtml(f"""
                    <div style="text-align: center; font-weight: bold; font-size: 36px;">天气播报</div>
                    <div style="margin: 15px; font-size: 30px;">
                        {weather_text}
                    </div>
                """)
            else:
                self.result_display.setHtml("""
                    <div style="text-align: center; font-weight: bold; font-size: 36px;">天气播报</div>
                    <div style="margin: 15px; color: red; font-size: 30px;">
                        获取天气信息失败，请稍后再试
                    </div>
                """)
        except Exception as e:
            self.result_display.setHtml(f"""
                <div style="text-align: center; font-weight: bold; font-size: 36px;">天气播报错误</div>
                <div style="margin: 15px; color: red; font-size: 30px;">
                    发生错误: {str(e)}
                </div>
            """)

    def toggle_cam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(DEFAULT_CAM_ID)
            if self.cap.isOpened():
                self.timer.start(33)
                self.button.setText("关闭摄像头")
                self.accumulate_progress = 0
                self.reset_display()
                self.last_frame_time = QDateTime.currentDateTime()
                self.last_char = ""
                self.current_sentence = ""
                self.recent_results = []
                # 重置进度跟踪变量
                self.last_progress = 0
                self.progress_unchanged_count = 0
                # 摄像头开启时禁用天气按钮
                self.weather_button.setEnabled(False)
            else:
                self.cap = None
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.button.setText("启动摄像头")
            self.current_sentence = ""
            self.recent_results = []
            self.last_char = ""
            self.reset_display()
            # 摄像头关闭时启用天气按钮
            self.weather_button.setEnabled(True)

    def update_frame(self):
        if not self.cap: return
        ret, frame = self.cap.read()
        if not ret: return
        img = self.co_helper.letter_box(frame.copy(), new_shape=IMG_SIZE[::-1], pad_color=(0, 0, 0))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outputs = self.model.run([img_rgb])
        boxes, classes, scores = post_process(outputs)
        vis_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detected_char = ""  # 当前检测到的字符
        conf = 0.0  # 当前置信度

        if boxes is not None and len(boxes) > 0:
            real_boxes = self.co_helper.get_real_box(boxes)

            # 找到当前帧中置信度最高的检测结果
            max_confidence = -1
            best_index = -1

            for i in range(len(scores)):
                if scores[i] > max_confidence:
                    max_confidence = scores[i]
                    best_index = i

            # 只识别置信度0.6以上的目标
            if best_index != -1 and scores[best_index] >= OBJ_THRESH:
                box = real_boxes[best_index]
                cls = classes[best_index]

                cls_code = CLASSES[cls]
                cls_cn = CLASS_CN_MAP.get(cls_code, cls_code)
                detected_char = cls_cn
                conf = scores[best_index]

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 只保留检测框，不显示文字

                # 添加到最近结果列表
                self.recent_results.append(cls_cn)

        # 记录当前检测到的字符
        self.last_char = detected_char

        # 更新进度条
        if self.cap and self.cap.isOpened():
            now = QDateTime.currentDateTime()
            ms_since_last = self.last_frame_time.msecsTo(now)
            self.last_frame_time = now

            # 如果有检测到字符则增加进度，否则缓慢减少
            if detected_char:
                # 上限改为1500毫秒（1.5秒）
                self.accumulate_progress = min(self.accumulate_progress + ms_since_last, RECOGNITION_TIME_THRESHOLD)
                self.progress_direction = 1
            else:
                # 缓慢减少进度，但不会低于0
                self.accumulate_progress = max(self.accumulate_progress - ms_since_last * 0.2, 0)
                self.progress_direction = -1

        # 绘制进度条 - 增大尺寸
        bar_height = 30  # 增大高度
        bar_width = vis_rgb.shape[1] - 40  # 增加宽度
        bar_y = vis_rgb.shape[0] - bar_height - 20  # 调整位置

        # 进度条背景 - 改为半透明灰色
        overlay = vis_rgb.copy()
        cv2.rectangle(overlay, (20, bar_y), (bar_width, bar_y + bar_height), (100, 100, 100), -1)
        alpha = 0.6  # 透明度
        cv2.addWeighted(overlay, alpha, vis_rgb, 1 - alpha, 0, vis_rgb)

        # 填充进度 - 改为蓝色半透明
        progress_width = int(bar_width * (self.accumulate_progress / RECOGNITION_TIME_THRESHOLD))
        if progress_width > 0:
            overlay = vis_rgb.copy()
            # 使用蓝色 (0, 0, 255) 并添加透明度
            cv2.rectangle(overlay, (20, bar_y), (20 + progress_width, bar_y + bar_height), (255, 150, 0), -1)
            alpha = 0.7  # 透明度
            cv2.addWeighted(overlay, alpha, vis_rgb, 1 - alpha, 0, vis_rgb)

        # 进度条边框 - 加粗
        cv2.rectangle(vis_rgb, (20, bar_y), (bar_width, bar_y + bar_height), (255, 255, 255), 3)

        # 调整图像大小以适应显示区域
        target_width = self.label.width()
        target_height = self.label.height()

        if target_width > 0 and target_height > 0:
            # 计算保持宽高比的缩放
            h, w, _ = vis_rgb.shape
            scale = min(target_width / w, target_height / h)
            new_width = int(w * scale)
            new_height = int(h * scale)

            # 缩放图像
            vis_rgb = cv2.resize(vis_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # 如果需要，在图像周围添加黑色边框以填满整个区域
            if new_width < target_width or new_height < target_height:
                border_top = (target_height - new_height) // 2
                border_bottom = target_height - new_height - border_top
                border_left = (target_width - new_width) // 2
                border_right = target_width - new_width - border_left

                vis_rgb = cv2.copyMakeBorder(
                    vis_rgb,
                    border_top, border_bottom,
                    border_left, border_right,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)
                )

        # 转换为QPixmap并显示
        h, w, ch = vis_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(vis_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.label.setPixmap(pix)

        # 更新右侧文本区域显示
        self.update_result_display()

    def update_result_display(self):
        # 构造右侧文本区域的显示内容
        # 当前识别句子显示为白色 - 使用更大的字体 (36px)
        current_sentence = f"<div style='font-size: 36px; color: #ffffff; margin: 15px;'>{self.current_sentence}</div>" if self.current_sentence else ""

        # 历史记录使用更大的字体
        history_html = ""
        if self.last_history:  # 如果有历史记录
            history_html = "<div style='margin-top: 20px; border-top: 2px solid #555; padding-top: 15px;'>"
            history_html += "<div style='font-weight: bold; font-size: 32px; margin-bottom: 15px;'>历史记录:</div>"
            history_html += f"<div style='font-size: 34px; margin: 10px; padding: 10px; background-color: rgba(50, 50, 50, 100); border-radius: 8px;'>{self.last_history}</div>"
            history_html += "</div>"

        # 进度条HTML（不再显示百分比文本）
        progress_percent = self.accumulate_progress / (RECOGNITION_TIME_THRESHOLD / 100)
        progress_bar = f"""
        <div style="margin-top: 20px;">
            <div style="background-color: rgba(50, 50, 50, 120); border: 2px solid gray; height: 35px; border-radius: 5px;">
                <div style="background-color: rgba(0, 150, 255, 150); 
                             width: {progress_percent}%; height: 100%; border-radius: 5px;"></div>
            </div>
        </div>
        """

        # 当前检测信息使用更大的字体 (30px)
        current_detection = f"<div style='font-size: 30px; margin-top: 20px;'>当前检测: {self.last_char if self.last_char else '无'}</div>"

        # 添加进度条状态信息
        progress_status = f"<div style='font-size: 30px; margin-top: 10px;'>进度条状态: {'识别中...' if self.progress_direction == 1 else '等待中...'}</div>"

        content = f"""
        <div style="text-align: center; font-weight: bold; font-size: 36px; margin-bottom: 20px;">手语翻译结果</div>
        <div style="margin: 20px;">
            {current_sentence}
            {progress_bar}
            {current_detection}
            {progress_status}
            {history_html}
        </div>
        """
        self.result_display.setHtml(content)

    def update_accumulated_result(self):
        if not self.cap: return

        # 当进度条满1500ms（1.5秒）时，处理累积的结果
        if self.accumulate_progress >= RECOGNITION_TIME_THRESHOLD and self.recent_results:
            # 找出最常出现的字符
            counter = Counter(self.recent_results)
            most_common, count = counter.most_common(1)[0]

            # 添加到当前句子
            self.current_sentence += most_common

            # 更新时间戳
            self.last_detect_time = QDateTime.currentDateTime()

            # 重置状态
            self.recent_results = []
            self.accumulate_progress = 0
            self.last_char = most_common

    def check_idle_and_output_sentence(self):
        if not self.cap: return

        # 检查进度条是否变化
        current_progress = self.accumulate_progress
        if abs(current_progress - self.last_progress) < 1:  # 如果进度基本没变
            self.progress_unchanged_count += 1
        else:
            self.progress_unchanged_count = 0  # 进度有变化，重置计数

        self.last_progress = current_progress  # 更新上次进度值

        # 如果当前有句子内容，并且进度条5秒没有变化（即停止增加）
        if self.current_sentence.strip() and self.progress_unchanged_count >= 1.5:
            sentence = self.current_sentence.strip()

            # 修改：只保留最近一句历史记录
            self.last_history = sentence

            data = {
                "device_id": "dev001",
                "temperature": sentence,
                "humidity": 21.2
            }

            requests.post("http://qedd8db8.natappfree.cc/upload", json=data)

            # 更新历史记录显示
            self.result_display.setHtml(f"""
                <div style="text-align: center; font-weight: bold; font-size: 36px; margin-bottom: 20px;">手语翻译结果</div>
                <div style="margin: 20px;">
                    <div style='font-size: 36px; color: #ffffff; margin: 15px;'></div>
                    {progress_bar if 'progress_bar' in locals() else ''}
                    <div style='font-size: 30px; margin-top: 20px;'>识别完成!</div>
                    <div style='font-size: 30px; margin-top: 10px;'>已保存句子</div>
                    <div style="margin-top: 20px; border-top: 2px solid #555; padding-top: 15px;">
                        <div style='font-weight: bold; font-size: 32px; margin-bottom: 15px;'>历史记录:</div>
                        <div style='font-size: 34px; margin: 10px; padding: 10px; background-color: rgba(50, 50, 50, 100); border-radius: 8px;'>{sentence}</div>
                    </div>
                </div>
            """)

            # 如果语音合成已启用，朗读这句话
            if self.tts_active:
                self.speak(sentence)

            # 重置当前句子和状态
            self.current_sentence = ""
            self.last_char = ""
            self.accumulate_progress = 0
            self.progress_unchanged_count = 0  # 重置计数

    def update_time(self):
        tz = QTimeZone(28800)
        now = QDateTime.currentDateTime().toTimeZone(tz)
        self.time_label.setText(f"北京时间: {now.toString('yyyy-MM-dd hh:mm:ss')}")

    def get_weather(self):
        try:
            url = "https://apis.tianapi.com/tianqi/index?key=3b3868b474ea6212d55cf5826ec8c711&city=101250101&type=1"
            r = requests.get(url, timeout=5)
            data = r.json().get("result", {})
            weather = data.get("weather", "N/A")
            temp = data.get("real", "--")
            icon = self.get_weather_icon(weather)
            self.weather_label.setText(f"{temp} {weather}")
            pix = QPixmap(50, 50)  # 增大图标尺寸
            pix.fill(Qt.transparent)
            painter = QPainter(pix)
            painter.setFont(QFont("Arial", 24, QFont.Bold))  # 增大图标字体
            painter.setPen(QColor(255, 165, 0))
            painter.drawText(pix.rect(), Qt.AlignCenter, icon)
            painter.end()
            self.weather_icon.setPixmap(pix)
        except Exception:
            self.weather_label.setText("天气获取失败")
            self.weather_icon.setText("⚠️")

    def get_weather_icon(self, txt):
        icons = {
            "晴": "☀️", "多云": "⛅", "阴": "☁️", "阵雨": "🌦", "雷阵雨": "⛈",
            "小雨": "🌧", "中雨": "🌧", "大雨": "🌧", "暴雨": "🌧",
            "雨夹雪": "🌨", "小雪": "❄️", "中雪": "❄️", "大雪": "❄️",
            "雾": "🌫", "霾": "😷"
        }
        return icons.get(txt, "🌈")

    def set_background(self, image_path):
        pixmap = QPixmap(image_path)
        if pixmap.isNull(): return
        scaled = pixmap.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        palette = self.palette()
        palette.setBrush(QPalette.Window, QBrush(scaled))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def resizeEvent(self, event):
        self.set_background(self._bg_path)
        super().resizeEvent(event)

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showFullScreen()
    sys.exit(app.exec())