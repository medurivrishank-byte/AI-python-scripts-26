import os
import time
import cv2
import numpy as np
import onnxruntime
HERE=os.path.dirname(os.path.abspath(__file__))
model=onnxruntime.InferenceSession(os.path.join(HERE,"modnet.onnx"))
INPUT=model.get_inputs()[0].name
FONT=cv2.FONT_HERSHEY_SIMPLEX

def cut_out(frame,size=256):
    height,width=frame.shape[:2]
    small=cv2.resize(frame,(round(width*size/height/32)*32,size))
    rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB).astype(np.float32)
    x=(rgb/127.5-1).transpose(2,0,1)[None]
    matte=model.run(None, {INPUT:x})[0][0,0]
    return cv2.resize(matte,(width,height))

def clean(matte):
    count, labels,stats,_=cv2.connectedComponentsWithStats((matte>0.35).astype(np.uint8))
    if count<2:
        return np.zeros_like(matte)
    biggest=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    outline,_=cv2.findContours((labels==biggest).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    body=cv2.drawContours(np.zeros_like(matte),outline,-1,1.0,-1)
    body=cv2.erode(body,np.ones((5,5),np.uint8))
    return cv2.GuassianBLur(body,(9,9),0)

def paste(scene, person, matte):
    alpha = matte[:, :, None]
    return (person * alpha + scene * (1 - alpha)).astype(np.uint8)

def final_photo(copies):
    newest_frame, newest_matte = copies[-1]
    photo = newest_frame
    for frame, matte in copies[:-1]:
        photo = paste(photo, frame, matte)
    return paste(photo, newest_frame, newest_matte)

video = cv2.VideoCapture(0)
copies = []
freeze_at = None
last_space = 0
message, message_until = "", 0

while True:
    ok, frame = video.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    matte = clean(cut_out(frame))

    if freeze_at and time.time() >= freeze_at:
        copies.append((frame, clean(cut_out(frame, 512))))
        print(f"Copy {len(copies)}: you fill {round(copies[-1][1].mean() * 100)}% of the picture")
        freeze_at = None

    view = frame.copy()
    for old_frame, old_matte in copies:
        view = paste(view, old_frame, old_matte)
    view = paste(view, frame, matte)

    if freeze_at:
        centre = (view.shape[1] // 2 - 40, view.shape[0] // 2 + 50)
        cv2.putText(view, str(int(freeze_at - time.time()) + 1), centre, FONT, 5, (255, 255, 255), 10)
    if time.time() < message_until:
        cv2.putText(view, message, (10, view.shape[0] - 15), FONT, 0.7, (0, 255, 255), 2)
    cv2.putText(view, f"{len(copies)} copies  SPACE freeze  s save  u undo  c clear  q quit", (10, 25), FONT, 0.6, (255, 255, 255), 2)
    cv2.imshow("Clone Me", view)

    code = cv2.waitKey(5)
    key = chr(code & 255).lower() if code != -1 else ""
    if key in ("q", "\x1b") or cv2.getWindowProperty("Clone Me", cv2.WND_PROP_VISIBLE) == 0:
        break
    if key == " ":
        if time.time() - last_space > 0.7 and not freeze_at:
            freeze_at = time.time() + 3
        last_space = time.time()
    if key == "u" and copies:
        copies.pop()
    if key == "c":
        copies.clear()
    if key == "s":
        if len(copies) < 2:
            message = "Freeze yourself at least 2 times first. Nothing saved."
        else:
            name = time.strftime("clone_photo_%H%M%S.jpg")
            cv2.imwrite(os.path.join(HERE, name), final_photo(copies))
            message = "Saved " + name
        message_until = time.time() + 2

video.release()
cv2.destroyAllWindows()
