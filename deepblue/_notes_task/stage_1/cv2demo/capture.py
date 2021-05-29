import cv2

# cap = cv2.VideoCapture('./data/video/butterfly.mp4')
cap = cv2.VideoCapture(0)
print('success')
while True:
    ret, frame = cap.read()
    cv2.imshow('arm', frame)
    if cv2.waitKey(1) == ord("q"):
#     if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
