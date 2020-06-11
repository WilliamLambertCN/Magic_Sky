import cv2
import matplotlib.pyplot as plt
import glob
import time

files = glob.glob('Demo/Videos/*.mp4')
# file = files[0]
for idx, file in enumerate(files):
    tic = time.time()
    print(file)
    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("fps %.2f, frame_num %d, size w & h %s" % (fps, frame_num, size))
    size = tuple(map(lambda x: int(x / 4), size))
    videowriter = cv2.VideoWriter('Demo/Modified/%s.mp4' % idx,
                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                  fps, (size[1], size[0]))  # 旋转后的size
    print("rotate & resize h %d w %d" % (size[0], size[1]))
    for i in range(int(frame_num)):
        success, frame = cap.read()
        assert success
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frame = cv2.rotate(frame, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        # plt.imshow(frame[..., ::-1])
        # plt.show()
        videowriter.write(frame)
    cap.release()
    videowriter.release()
    print(" %.2f s" % (time.time() - tic))
