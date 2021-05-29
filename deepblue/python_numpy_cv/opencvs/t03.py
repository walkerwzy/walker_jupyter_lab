import cv2

cap = cv2.VideoCapture("./aa.mp4")
print("success")
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("arm", gray)
    if cv2.waitKey(10) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
