import cv2

cap = cv2.VideoCapture("./aa.mp4")
print("success")
i = 0
while i < 150:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("arm1", frame)
    cv2.imwrite(f"./info2/{i}.jpg", frame)
    if cv2.waitKey(60) == ord("q"):
        break
    i += 1

cap.release()
cv2.destroyAllWindows()
