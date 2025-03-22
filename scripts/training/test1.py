import time
import pyautogui

def main():
    print("脚本将每隔4秒自动输入一次 'a'。请将光标聚焦在需要输入的窗口。")
    time.sleep(5)  # 给你时间切换到目标窗口
    
    while True:
        pyautogui.typewrite('a')
        time.sleep(5.5)

if __name__ == "__main__":
    main()
